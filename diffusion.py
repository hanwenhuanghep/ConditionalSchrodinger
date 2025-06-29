import math
import numpy as np
import torch

from misc import append_dims

bmin=1
bmax=1
sigma_min=1
sigma_max=10

class Follmer:

    def __init__(self, args):
        self.args = args

    def get_alpha(self, t):
        return t
    
    def beta_fun(self, t):
        return bmin+t*(bmax-bmin)

    def beta_int_fun(self, t):
        return bmin*t+0.5*(bmax-bmin)*t**2

    def xi_fun(self, t):
        return torch.exp(-torch.tensor(0.5)*(self.beta_int_fun(1)-self.beta_int_fun(t)))

    def tau_fun(self, t):
        return torch.exp(-torch.tensor(0.5)*(self.beta_int_fun(1)-self.beta_int_fun(0)))

    def sigma1_fun(self, t):
        tau=self.tau_fun(t)
        return 1-tau**2
    
    def sigma2_fun(self, t):
        xi=self.xi_fun(t)
        return 1-xi**2

    def sigmas_fun(self, t):
        sigma1=self.sigma1_fun(t)
        sigma2=self.sigma2_fun(t)
        return sigma1*sigma2/(sigma1-sigma2) 
    
    def sigma_fun(self, t):
        return sigma_min**2*(sigma_max/sigma_min)**(2*t)
    
    def dsigma_fun(self, t):
        return 2*self.sigma_fun(t)*torch.log(torch.tensor(sigma_max/sigma_min))
    
    def get_alphac(self, t):
        return 1-t
    
    def get_dalpha(self, t):
        return torch.ones_like(t)

    def get_beta(self, t):
        return (t*(2-t)).sqrt()
        
    def get_beta2(self, t):
        return t*(2-t)

    def sampling_prior(self, shape, device):
        return torch.zeros_like(shape, device=device)
    
    def sampling_priorc(self, shape, device):
        return torch.randn(shape, device=device)
    
    def get_cnoise(self, t):
        return self.args.M*t
    
    def get_cin(self, t):
        alpha = self.get_alphac(t)
        beta2 = self.get_beta2(t)
        return 1/((alpha*self.args.sigma_data)**2+beta2).sqrt()
    
    def get_cout(self, t, s=None):
        alpha = self.get_alphac(t)
        beta = self.get_beta(t)
        beta2 = self.get_beta2(t)
        if s is not None:
            return (t-s)/beta2
        else:
            return beta*self.args.sigma_data / ((alpha*self.args.sigma_data)**2+beta2).sqrt()
    
    def get_cskip(self, t, s=None):
        alpha = self.get_alphac(t)
        beta2 = self.get_beta2(t)
        if s is not None:
            return 1 + alpha*(s-t)/beta2
        else:
            return alpha*self.args.sigma_data**2 / ((alpha*self.args.sigma_data)**2+beta2)
    
    def get_weightning(self, t):
        alpha = self.get_alphac(t)
        beta2 = self.get_beta2(t)
        return ((alpha*self.args.sigma_data)**2+beta2) / beta2*self.args.sigma_data**2
    
    def get_denoiser(self, model, x, t, cond, s=None):
        ndim = x.ndim
        cnoise = self.get_cnoise(t)
        cnoise_s = self.get_cnoise(s) if s is not None else None
        cin = self.get_cin(t)
        cout = self.get_cout(t)
        cskip = self.get_cskip(t)
        return append_dims(cskip, ndim)*x + append_dims(cout, ndim)*model(append_dims(cin, ndim)*x, cnoise, cond, cnoise_s)
    
    def get_forward_drift(self, model, x, t, cond):
        xt=model(x,t,cond)
        tmp=(xt-x)/(1-t[0])
        return tmp

    def get_alpha_drift(self, model, x, t, cond):
        xt=model(x,t,cond)
        tmp=self.dsigma_fun(t[0])*(xt-x)/(self.sigma_fun(1)-self.sigma_fun(t[0]))
        return tmp

    def get_beta_drift(self, model, x, t, cond):
        xt=model(x,t,cond)
        beta=self.beta_fun(t[0])
        xi=self.xi_fun(t[0])
        tmp=beta*xi*(xt-xi*x)/(1-xi**2)
        return tmp

    def get_velocity(self, model, x, t, cond):
        ndim = x.ndim
        alpha = self.get_alphac(t)
        beta2 = self.get_beta2(t)
        return (append_dims(alpha, ndim)*x - self.get_denoiser(model, x, t, cond)) / append_dims(beta2, ndim)
      
    def get_forward_diffusion(self, t,device):        
        return self.get_dalpha(t).sqrt()

    def compute_schr_loss(self, model, batch, cond):
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device
        #randomly select the time points
        t_long = torch.randint(0, self.args.num_steps, (bsz, ), device=device).float()
        dt = (1 - self.args.eps0 - self.args.eps1) / self.args.num_steps
        t = self.args.eps0 + t_long*dt
        alphas = t
        sigmas=t*(1-t)
        sigmas=torch.sqrt(sigmas)
        #generated xt
        noised = append_dims(alphas, ndim)*batch + append_dims(sigmas, ndim)*torch.randn_like(batch, device=device)
        denoised = model(noised, t, cond)
        loss = (denoised-batch)**2
        return torch.mean(loss)

    def compute_scha_loss(self, model, batch, cond):
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device
        t_long = torch.randint(0, self.args.num_steps, (bsz, ), device=device).float()
        dt = (1 - self.args.eps0 - self.args.eps1) / self.args.num_steps
        t = self.args.eps0 + t_long*dt
        alphas = (self.sigma_fun(t)-self.sigma_fun(0))/(self.sigma_fun(1)-self.sigma_fun(0))
        sigmas=(self.sigma_fun(t)-self.sigma_fun(0))*(self.sigma_fun(1)-self.sigma_fun(t))/(self.sigma_fun(1)-self.sigma_fun(0))
        sigmas=torch.sqrt(sigmas)
        noised = append_dims(alphas, ndim)*batch + append_dims(sigmas, ndim)*torch.randn_like(batch, device=device)
        denoised = model(noised, t, cond)
        loss = (batch-denoised)**2
        return torch.mean(loss)
    
    def compute_schb_loss(self, model, batch, cond):
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device
        t_long = torch.randint(0, self.args.num_steps, (bsz, ), device=device).float()
        dt = (1 - self.args.eps0 - self.args.eps1) / self.args.num_steps
        t = self.args.eps0 + t_long*dt
        alphas = self.sigma2_fun(t)/self.sigmas_fun(t)/self.xi_fun(t)
        sigmas=1/self.sigmas_fun(t)
        sigmas=self.sigma2_fun(t)*torch.sqrt(sigmas)/self.xi_fun(t)
        noised = append_dims(alphas, ndim)*batch + append_dims(sigmas, ndim)*torch.randn_like(batch, device=device)
        denoised = model(noised, t, cond)
        loss = (batch-denoised)**2
        return torch.mean(loss)
    
    def solve_sde(self, model, x, grid, solver, cond):
        bsz = x.shape[0]
        device = x.device
        xt = x
        steps = len(grid) if grid.ndim == 1 else grid.shape[1]
        for i in range(steps-1):
            if grid.ndim == 1:
                t = torch.ones(bsz, device=device)*grid[i]
                dt = grid[i+1] - grid[i]
            elif grid.ndim == 2:
                t = grid[:, i]
                dt = grid[:, i+1] - grid[:, i]
            else:
                raise ValueError(f"unsupported grid shape:{grid.shape}")
            if solver.lower() == "euler-maruyama":
                xt = self.euler_maruyama_step(model, xt, t, cond, dt)
            elif solver.lower() == "alpha-maruyama":
                xt = self.alpha_maruyama_step(model, xt, t, cond, dt)
            elif solver.lower() == "beta-maruyama":
                xt = self.beta_maruyama_step(model, xt, t, cond, dt)
            else:
                raise ValueError(f"unsupported solver{self.args.sde_solver}")
        return xt
    
    def euler_maruyama_step(self, model, x, t, cond, dt):
        ndim = x.ndim
        device = x.device
        dt_sqrt = math.sqrt(math.abs(dt)) if isinstance(dt, int) else dt.abs().sqrt()  
        tmp1=self.get_forward_drift(model, x, t, cond)*dt 
        tmp2=dt_sqrt*torch.randn_like(x, device=device)
        tmp=x+tmp1+tmp2  
        return tmp

    def alpha_maruyama_step(self, model, x, t, cond, dt):
        ndim = x.ndim
        device = x.device
        dt_sqrt = math.sqrt(math.abs(dt)) if isinstance(dt, int) else dt.abs().sqrt()  
        dsigma=self.dsigma_fun(t[0])
        tmp1=self.get_alpha_drift(model, x, t, cond)*dt 
        tmp2=torch.sqrt(dsigma)*dt_sqrt*torch.randn_like(x, device=device)
        tmp=x+tmp1+tmp2  
        return tmp

    def beta_maruyama_step(self, model, x, t, cond, dt):
        ndim = x.ndim
        device = x.device
        dt_sqrt = math.sqrt(math.abs(dt)) if isinstance(dt, int) else dt.abs().sqrt()  
        dsigma=self.beta_fun(t[0])
        tmp1=(self.get_beta_drift(model, x, t, cond)-0.5*dsigma*x)*dt 
        tmp2=torch.sqrt(dsigma)*dt_sqrt*torch.randn_like(x, device=device)
        tmp=x+tmp1+tmp2  
        return tmp

    def compute_dsm_loss(self, model, batch, cond):
        bsz = batch.shape[0]
        ndim = batch.ndim
        device = batch.device
        t_long = torch.randint(0, self.args.num_steps, (bsz, ), device=device).float()
        dt = (1 - self.args.eps0 - self.args.eps1) / self.args.num_steps
        t = self.args.eps0 + t_long*dt
        alpha = self.get_alphac(t)
        beta = self.get_beta(t)
        weightnings = self.get_weightning(t)
        noised = append_dims(alpha, ndim)*batch + append_dims(beta, ndim)*torch.randn_like(batch, device=device)
        denoised = self.get_denoiser(model, noised, t, cond)
        loss = append_dims(weightnings, ndim)*(denoised - batch)**2
        return torch.mean(loss)

    def solve_ode(self, model, x, grid, solver, cond):
        """Solve system starting at x from grid[0] to grid[-1], or grid[:, 0] to grid[:, -1]."""
        bsz = x.shape[0]
        device = x.device
        xt = x
        steps = len(grid) if grid.ndim == 1 else grid.shape[1]
        for i in range(steps-1):
            if grid.ndim == 1:
                t = torch.ones(bsz, device=device)*grid[i]
                dt = grid[i+1] - grid[i]
            elif grid.ndim == 2:
                t = grid[:, i]
                dt = grid[:, i+1] - grid[:, i]
            else:
                raise ValueError(f"unsupported grid shape:{grid.shape}")
            if solver.lower() == "euler":
                xt = self.euler_step(model, xt, t, cond, dt)
            elif solver.lower() == "heun":
                xt = self.heun_step(model, xt, t, cond, dt)
            else:
                raise ValueError(f"unsupported solver{self.args.ode_solver}")
        return xt

    def euler_step(self, model, x, t, cond, dt):
        return x + self.get_velocity(model, x, t, cond)*dt

