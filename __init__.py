import os
import numpy
import scipy.sparse
import scipy as sp
import tqdm
import torch
import sklearn
import numpy.random as npr
import types
import numpy as np
import tqdm

def calc_Zs(cells,genes,U,alpha,batchsize=5000,verbose=True):
    '''
    By batching we avoid ever allocating an array of size "N_nonzero x Nk"
    '''
    Z=torch.zeros_like(cells,dtype=U.dtype)
    bins=np.r_[0:len(cells):batchsize,len(cells)]
    rg=range(len(bins)-1)
    if verbose:
        rg=tqdm.tqdm_notebook(rg,leave=False)
    for i in rg:
        cells2=cells[bins[i]:bins[i+1]]
        genes2=genes[bins[i]:bins[i+1]]
        Z2=torch.sum(U[cells2]*alpha[genes2],1)
        Z[bins[i]:bins[i+1]]=Z2
    return Z

def simple_log_likelihood(cells,genes,U,alpha,batchsize=5000):
    Z=calc_Zs(cells,genes,U,alpha,batchsize=batchsize)
    return torch.sum(torch.log(Z))-torch.sum(torch.sum(U,0)*torch.sum(alpha,0))

def gamma_kl(alphap,betap,alphaq,betaq):
    T1=(alphap-alphaq)*torch.digamma(alphap)
    T2=-torch.lgamma(alphap)+torch.lgamma(alphaq)
    T3=alphaq*(torch.log(betap)-torch.log(betaq))
    T4=alphap*(betaq-betap)/betap

    rez=T1+T2+T3+T4

    return rez

def accumulate_multiplier_batched(accumulator,cells,genes,vals,elogU,elogalpha,batchsize=500,verbose=True):
    '''
    By batching we avoid ever allocating an array of size "N_nonzero x Nk"
    '''
    
    bins=np.r_[0:len(cells):batchsize,len(cells)]
    
    rg=range(len(bins)-1)
    if verbose:
        rg=tqdm.tqdm_notebook(rg,leave=False)
        
    # batch it up
    for i in rg:
        cells2=cells[bins[i]:bins[i+1]]
        genes2=genes[bins[i]:bins[i+1]]
        vals2=vals[bins[i]:bins[i+1]]

        varphi=torch.softmax(elogU[cells2]+elogalpha[genes2],1) # nnz x Nk
        xi=vals2[:,None]* varphi # nnz x Nk

        accumulator.index_put_((cells2,),xi,accumulate=True)

def accumulate_ELBO(cells,genes,vals,uai,sU,rU,salpha,ralpha,kappas,lambdas,rhoU,rhoalpha,batchsize=500,verbose=True):
    # KLs
    U_kls = torch.sum(gamma_kl(sU,rU,rhoU,kappas[:,None]))
    alpha_kls = torch.sum(gamma_kl(salpha,ralpha,rhoalpha,lambdas[:,None]))
       
    # Z term
    ll=0
    bins=np.r_[0:len(cells):batchsize,len(cells)]
    rg=range(len(bins)-1)
    if verbose:
        rg=tqdm.tqdm_notebook(rg,leave=False) 
    for i in rg:
        cells2=cells[bins[i]:bins[i+1]]
        genes2=genes[bins[i]:bins[i+1]]
        vals2=vals[bins[i]:bins[i+1]]

        explog = torch.exp(uai.ELU[cells2] + uai.ELalpha[genes2]) # nnz x k
        poislogterm=torch.log(torch.sum(explog,1)) # nnz
        ll += torch.sum(vals2*poislogterm)
    ll -= torch.sum(torch.sum(uai.EU,0)*torch.sum(uai.Ealpha,0))

    rez=types.SimpleNamespace(U_kls=U_kls.item(),alpha_kls=alpha_kls.item(),Z=ll.item())
    rez.total=rez.Z-rez.alpha_kls-rez.U_kls
    rez.loss=-rez.total


    return rez


def update_U(cells,genes,vals,uai,kappas,rho,verbose=True,batchsize=500):
    elogU = uai.ELU 
    elogalpha = uai.ELalpha 
    alphameans = uai.Ealpha 
    new_rU = kappas[:,None] + torch.sum(alphameans,0,keepdim=True)  # NC x 1  + 1 x NK
    new_sU = torch.full_like(uai.EU,rho)

    accumulate_multiplier_batched(new_sU,cells,genes,vals,elogU,elogalpha,verbose=verbose,batchsize=batchsize)

    return new_sU,new_rU

class UAlphaInfo:
    def __init__(self,EU,ELU,Ealpha,ELalpha):
        self.EU=EU
        self.ELU=ELU
        self.Ealpha=Ealpha
        self.ELalpha=ELalpha

    def transpose(self):
        return UAlphaInfo(self.Ealpha,self.ELalpha,self.EU,self.ELU)

class Trainer:
    def __init__(self,model,testingdata=None,test_iters=10):
        self.model=model
        self.elbos=[]
        self.elbos.append(self.model.ELBO(verbose=False))

        self.test_elbos=[]
        self.test_elbo_iter_loc=[]
        self.testingdata=testingdata
        self.test_iters=test_iters
        self.testcheck()

        self.KEEPGOING=False

    def _eupdate(self,t):
        self.elbos.append(self.model.ELBO(verbose=False))
        t.set_description("%e"%self.elbos[-1].loss) 
        if self.elbos[-1].total < self.elbos[0].total:
            self.KEEPGOING=False
            raise Exception("elbo went down!")

    def run_test(self,test_iters):
        params=self.model.save_params()
        test_model=PoisModel(self.testingdata,self.model.Nk,
            batchsize=self.model.batchsize,
            rhoU=self.model.rhoU.item(),
            rhoalpha=self.model.rhoalpha.item())
        test_model.initialize_with_known_alpha(params.salpha,params.ralpha)
        test_model.double()
        test_model.cuda()

        for i in range(self.test_iters):
            test_model.update_U(verbose=False)
            test_model.update_kappas()

        return test_model

    def run_train_scratch(self,test_iters):
        params=self.model.save_params()
        train_scratch_model=PoisModel(self.model.countmatrix,self.model.Nk,
            batchsize=self.model.batchsize,
            rhoU=self.model.rhoU.item(),
            rhoalpha=self.model.rhoalpha.item())
        train_scratch_model.initialize_with_known_alpha(params.salpha,params.ralpha)
        train_scratch_model.double()
        train_scratch_model.cuda()

        for i in range(self.test_iters):
            train_scratch_model.update_U(verbose=False)
            test_model.update_kappas()

        return train_scratch_model

    def testcheck(self,test_iters=None):
        if self.testingdata is not None:
            # get data
            self.test_model=self.run_test(self.test_iters)

            # get elbo, correcting for the fact that we 
            # have different number of cells our two datasets:
            rez=self.test_model.ELBO()
            ntrain=self.model.countmatrix.shape[0]
            ntest=self.test_model.countmatrix.shape[0]
            rez.total=rez.Z*(ntrain/ntest)-rez.alpha_kls-rez.U_kls*(ntrain/ntest)
            rez.loss=-rez.total
            self.test_elbos.append(rez)
            self.test_elbo_iter_loc.append(len(self.elbos)-1)
            # if len(self.test_elbos)>2 and self.test_elbos[-2].loss < self.test_elbos[-1].loss:
            #     raise Exception("test loss going up!")

    def go(self,niter,U=True,alpha=True,kappalams=True,testcheck_every=None):
        self.KEEPGOING=True

        if testcheck_every is None:
            testcheck_every=niter*2 # never checks
        else:
            assert self.testingdata is not None

        t=tqdm.tqdm_notebook(range(niter))
        for i in t:
            if U:
                self.model.update_U(verbose=False)
                self._eupdate(t)
            if alpha:
                self.model.update_alpha(verbose=False)
                self._eupdate(t)
            if kappalams:
                self.model.update_kappas_and_lambdas()
                self._eupdate(t)

            if (i+1)%testcheck_every ==0 :
                self.testcheck()

            if not self.KEEPGOING:
                return

        

class PoisModel:
    def __init__(self,countmatrix,Nk,dtype='float',rhoU=1.0,rhoalpha=1.0,batchsize=5000):
        self.Nk=Nk
        self.countmatrix=countmatrix

        self.Nc,self.Nt=self.countmatrix.shape
        self.Nk=Nk
        self.batchsize=batchsize

        if dtype=='float':
            self.dtype=torch.float
        else:
            self.dtype=torch.double
        self.rhoU=torch.tensor(rhoU,dtype=self.dtype)
        self.rhoalpha=torch.tensor(rhoalpha,dtype=self.dtype)

        coo=countmatrix.tocoo()
        self.cells=torch.tensor(coo.row,dtype=torch.long)
        self.genes=torch.tensor(coo.col,dtype=torch.long)
        self.vals=torch.tensor(coo.data,dtype=self.dtype)

    def reinitialize_Us(self):
        mn = len(self.cells) / (self.Nc*self.salpha.shape[1])
        avg = np.sqrt(mn / self.salpha.shape[1])

        self.sU=torch.full((self.Nc,self.Nk),float(avg),device=self.salpha.device,dtype=self.dtype)
        self.rU=torch.ones((self.Nc,self.Nk),device=self.salpha.device,dtype=self.dtype)
        
        self.update_uai()
        self.update_kappas_and_lambdas()

    def update_uai(self):
        EU = self.sU/self.rU
        ELU = torch.digamma(self.sU)-torch.log(self.rU)
        Ealpha = self.salpha/self.ralpha
        ELalpha = torch.digamma(self.salpha) - torch.log(self.ralpha)
        self.uai=UAlphaInfo(EU,ELU,Ealpha,ELalpha)

    def save_params(self):
        rez=types.SimpleNamespace()
        for x in ['kappas','lambdas','sU','rU','salpha','ralpha','rhoU','rhoalpha']:
            setattr(rez,x,getattr(self,x).detach().cpu().numpy())
        return rez

    def load_params(self,rez,device='cpu'):
        for x in ['kappas','lambdas','sU','rU','salpha','ralpha','rhoU','rhoalpha']:
            setattr(self,x,torch.tensor(getattr(rez,x),dtype=self.dtype,device=device))
        self.update_uai()

    def cuda(self):
        for x in ['cells','genes','kappas','lambdas','sU','rU','salpha','ralpha','rhoU','rhoalpha','vals']:
            setattr(self,x,getattr(self,x).cuda())
        self.update_uai()

    def double(self):
        self.dtype=torch.double
        for x in ['kappas','lambdas','sU','rU','salpha','ralpha','rhoU','rhoalpha','vals']:
            setattr(self,x,getattr(self,x).double())
        self.update_uai()

    def ELBO(self,verbose=True):
        return accumulate_ELBO(self.cells,self.genes,self.vals,
            self.uai,self.sU,self.rU,self.salpha,self.ralpha,self.kappas,self.lambdas,self.rhoU,self.rhoalpha,
            batchsize=self.batchsize,verbose=verbose)

    def update_U(self,verbose=True):
        self.sU,self.rU=update_U(self.cells,self.genes,self.vals,self.uai,
            self.kappas,verbose=verbose,batchsize=self.batchsize,
            rho=self.rhoU)
        self.uai.EU = self.sU/self.rU
        self.uai.ELU = torch.digamma(self.sU)-torch.log(self.rU)        

    def update_alpha(self,verbose=True):
        self.salpha,self.ralpha=update_U(self.genes,self.cells,self.vals,self.uai.transpose(),
            self.lambdas,verbose=verbose,batchsize=self.batchsize,
            rho=self.rhoalpha)
        self.uai.Ealpha = self.salpha/self.ralpha
        self.uai.ELalpha = torch.digamma(self.salpha)-torch.log(self.ralpha)        

    def update_kappas(self):
        self.kappas = self.rhoU /torch.mean(self.uai.EU,1)
    def update_lambdas(self):
        self.lambdas= self.rhoalpha /torch.mean(self.uai.Ealpha,1)
    def update_kappas_and_lambdas(self):
        self.kappas = self.rhoU /torch.mean(self.uai.EU,1)
        self.lambdas= self.rhoalpha /torch.mean(self.uai.Ealpha,1)

    def initialize_with_known_alpha(self,salpha,ralpha,device='cpu',verbose=True):
        mn = np.mean(self.countmatrix)        
        avg = np.sqrt(mn / self.Nk)

        self.salpha=torch.tensor(salpha,dtype=self.dtype,device=device)
        self.ralpha=torch.tensor(ralpha,dtype=self.dtype,device=device)
        self.sU=torch.full((self.Nc,self.Nk),avg,dtype=self.dtype)
        self.rU=torch.ones_like(self.sU, device=device)
        
        self.update_uai()
        self.update_kappas_and_lambdas()

    def initialize_with_known_U(self,sU,rU,device='cpu',verbose=True):
        mn = np.mean(self.countmatrix)
        avg = np.sqrt(mn / self.Nk)

        self.sU=torch.tensor(sU,dtype=self.dtype,device=device)
        self.rU=torch.tensor(rU,dtype=self.dtype,device=device)
        self.salpha=torch.full((self.Nt,self.Nk),avg,dtype=self.dtype)
        self.ralpha=torch.ones_like(self.salpha, device=device)
        
        self.update_uai()
        self.update_kappas_and_lambdas()

    def initialize_randomly(self,device='cpu',certainty=10,startwith='U',verbose=True):
        if startwith=='U':
            cellsums=np.array(np.mean(self.countmatrix,axis=1))
            mns=npr.rand(self.Nc,self.Nk)
            mns=np.sqrt(cellsums)*mns/np.sum(mns,axis=1,keepdims=True)
            self.initialize_with_known_U(certainty+0*mns,certainty/mns,verbose=verbose)
        else:            
            cellsums=np.array(np.mean(self.countmatrix,axis=0))
            mns=npr.rand(self.Nt,self.Nk)
            mns=np.sqrt(cellsums)*mns/np.sum(mns,axis=1,keepdims=True)
            self.initialize_with_known_alpha(certainty+0*mns,certainty/mns,verbose=verbose)

    def initialize(self,device='cpu',certainty=10):
        U,alpha=sklearn.decomposition.nmf._initialize_nmf(self.countmatrix.tocsr(),self.Nk)

        alpha=alpha.T
        avg = self.countmatrix.mean()
        U[U == 0] = abs(avg * npr.randn(len(U[U == 0])) / 100)
        alpha[alpha == 0] = abs(avg *npr.randn(len(alpha[alpha == 0])) / 100)
        
        self.sU=certainty*torch.tensor(U,dtype=self.dtype,device=device)
        self.rU=certainty*torch.ones_like(self.sU)
        self.salpha=certainty*torch.tensor(alpha,dtype=self.dtype,device=device)
        self.ralpha=certainty*torch.ones_like(self.salpha)
        
        self.update_uai()
        self.update_kappas_and_lambdas()
