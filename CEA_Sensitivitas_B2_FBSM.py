import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

OUT=Path('hasil_CEA_sensitivitas_B2'); OUT.mkdir(exist_ok=True)
P=288315899.0
X0=np.array([288016688,100000,75690,70796,50000,1000,941,784],float)
p=dict(beta_s=.07647,beta_r=.07171,sigma_s=.009319,sigma_r=.00846,tau_s=.05909,tau_r=.01414,kappa=.9863,rho=.01523,mu=.000269,delta_s=.000439,delta_r=.00365,gamma_s=1/26,gamma_r=.0075,omega=.00278,eta=.51)
p['Lambda']=p['mu']*P
T=156.; tgrid=np.linspace(0,T,157)
A1,A2,B1=1.,20.,1e5
U1MAX,U2MAX=.8,1.

def rhs(y,u1,u2):
    S,Es,Is,Ts,R,Er,Ir,Tr=y
    N=np.sum(y)
    ls=(1-u1)*p['beta_s']*Is/N
    lr=(1-u1)*p['beta_r']*Ir/N
    kap=p['kappa']+(1-p['kappa'])*u2
    mu=p['mu']; gs=p['gamma_s']; gr=p['gamma_r']; rho=p['rho']; eta=p['eta']
    return np.array([
        p['Lambda']-(ls+lr)*S-mu*S+p['omega']*R,
        ls*(S+eta*R)-(p['sigma_s']+mu)*Es,
        p['sigma_s']*Es+gs*(1-rho)*(1-kap)*Ts-(p['tau_s']+mu+p['delta_s'])*Is,
        p['tau_s']*Is-(gs+mu)*Ts,
        gs*kap*Ts+gr*Tr-eta*(ls+lr)*R-(p['omega']+mu)*R,
        lr*(S+eta*R)-(p['sigma_r']+mu)*Er,
        p['sigma_r']*Er+gs*rho*(1-kap)*Ts-(p['tau_r']+mu+p['delta_r'])*Ir,
        p['tau_r']*Ir-(gr+mu)*Tr
    ])

def jacobian(y,u1,u2):
    h=1e-20; J=np.zeros((8,8))
    for j in range(8):
        yc=np.asarray(y,dtype=complex).copy(); yc[j]+=1j*h
        J[:,j]=np.imag(rhs(yc,u1,u2))/h
    return J

def forward(u1,u2):
    def f(t,y):return rhs(y,np.interp(t,tgrid,u1),np.interp(t,tgrid,u2))
    sol=solve_ivp(f,(0,T),X0,method='LSODA',t_eval=tgrid,dense_output=True,rtol=1e-7,atol=1e-5)
    if not sol.success:raise RuntimeError(sol.message)
    return sol

def backward(state,u1,u2):
    grad=np.array([0,0,A1,0,0,0,A2,0],float)
    def f(t,lam):
        y=state.sol(t); a=np.interp(t,tgrid,u1); b=np.interp(t,tgrid,u2)
        return -grad-jacobian(y,a,b).T@lam
    sol=solve_ivp(f,(T,0),np.zeros(8),method='LSODA',t_eval=tgrid[::-1],rtol=1e-6,atol=1e-5)
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,::-1]

def controls(states,adj,B2):
    u1=np.zeros(len(tgrid));u2=np.zeros(len(tgrid))
    for i in range(len(tgrid)):
        y=states[:,i];lam=adj[:,i]
        S,Es,Is,Ts,R,Er,Ir,Tr=y;N=np.sum(y)
        bs=p['beta_s']*Is/N;br=p['beta_r']*Ir/N
        d1=np.zeros(8);d1[0]=(bs+br)*S;d1[1]=-bs*(S+p['eta']*R);d1[4]=p['eta']*(bs+br)*R;d1[5]=-br*(S+p['eta']*R)
        d2=np.zeros(8);scale=p['gamma_s']*(1-p['kappa'])*Ts
        d2[2]=-(1-p['rho'])*scale;d2[4]=scale;d2[6]=-p['rho']*scale
        u1[i]=np.clip(-np.dot(lam,d1)/B1,0,U1MAX)
        u2[i]=np.clip(-np.dot(lam,d2)/B2,0,U2MAX)
    return u1,u2

def solve(strategy,B2):
    u1=np.full(len(tgrid),.4 if strategy in ('B','D') else 0.)
    u2=np.full(len(tgrid),.2 if strategy in ('C','D') else 0.)
    if strategy=='A':return forward(u1,u2),u1,u2,True,0
    tol=1e-5;relax=.3;maxiter=300
    converged=False
    for it in range(maxiter):
        state=forward(u1,u2)
        adj=backward(state,u1,u2)
        v1,v2=controls(state.y,adj,B2)
        if strategy=='B':v2[:]=0
        if strategy=='C':v1[:]=0
        nu1=(1-relax)*u1+relax*v1;nu2=(1-relax)*u2+relax*v2
        err=max(np.max(np.abs(nu1-u1)),np.max(np.abs(nu2-u2)))
        u1,u2=nu1,nu2
        if err<tol:converged=True;break
    return forward(u1,u2),u1,u2,converged,it+1

records=[]; control_records=[]; baselines={}
for B2 in [3000.,30000.,300000.]:
    print('\nB2 =',B2,flush=True)
    res={}
    for s in ['A','B','C','D']:
        sol,u1,u2,ok,nit=solve(s,B2)
        burden=np.trapezoid(A1*sol.y[2]+A2*sol.y[6],tgrid)
        cost=np.trapezoid(.5*B1*u1*u1+.5*B2*u2*u2,tgrid)
        res[s]=dict(sol=sol,u1=u1,u2=u2,burden=burden,cost=cost,ok=ok)
        print(s,'converged',ok,'iterations',nit,'burden',round(burden,1),'cost',round(cost,1),flush=True)
    baseline=res['A']['burden']
    for s,r in res.items():
        eff=baseline-r['burden'];cost=r['cost']
        records.append(dict(B2=B2,Strategy=s,Converged=r['ok'],Cost=cost,Effectiveness=eff,ACER=(cost/eff if eff>0 else np.nan),Is_156=r['sol'].y[2,-1],Ir_156=r['sol'].y[6,-1],Burden=r['burden'],Objective_J=r['burden']+cost,Mean_u1=np.mean(r['u1']),Mean_u2=np.mean(r['u2'])))
        for t,a,b in zip(tgrid,r['u1'],r['u2']):control_records.append(dict(B2=B2,Strategy=s,Week=t,u1=a,u2=b))

df=pd.DataFrame(records);df.to_csv(OUT/'CEA_sensitivity_B2_summary.csv',index=False)
pd.DataFrame(control_records).to_csv(OUT/'CEA_sensitivity_B2_controls.csv',index=False)
rows=[]
for B2,g in df.groupby('B2'):
    z=g.set_index('Strategy')
    for hi,lo in [('B','A'),('C','A'),('D','A'),('D','B')]:
        dc=z.loc[hi,'Cost']-z.loc[lo,'Cost'];de=z.loc[hi,'Effectiveness']-z.loc[lo,'Effectiveness']
        rows.append(dict(B2=B2,Comparison=hi+' vs '+lo,Delta_Cost=dc,Delta_Effect=de,ICER=(dc/de if abs(de)>1e-9 else np.nan)))
icers=pd.DataFrame(rows);icers.to_csv(OUT/'CEA_sensitivity_B2_ICER.csv',index=False)
print('\nSUMMARY\n',df.to_string(index=False));print('\nICER\n',icers.to_string(index=False))

fig,axs=plt.subplots(1,2,figsize=(12,5))
for s in ['B','C','D']:
    g=df[df.Strategy==s]
    axs[0].plot(g.B2,g.ACER,'-o',label='Strategy '+s)
    axs[1].plot(g.B2,g.Effectiveness/1e6,'-o',label='Strategy '+s)
axs[0].set_ylabel('ACER (modeled cost / weighted person-week)')
axs[1].set_ylabel('Weighted infectious person-weeks averted (millions)')
for ax in axs:
    ax.set_xscale('log');ax.set_xlabel(r'Adherence cost weight $B_2$');ax.grid(alpha=.3);ax.legend()
plt.tight_layout();plt.savefig(OUT/'Figure_CEA_Sensitivity_B2.png',dpi=300,bbox_inches='tight');plt.savefig(OUT/'Figure_CEA_Sensitivity_B2.pdf',bbox_inches='tight');plt.close()
print('\nSaved to',OUT.resolve())
