from mpl_toolkits.axes_grid.inset_locator import inset_axes
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pymultinest
from scipy import interpolate
import numpy as np
import argparse
import utils
import os

parser = argparse.ArgumentParser()
parser.add_argument('-lc',default='')
parser.add_argument('-lc_to_detrend',default='')

args = parser.parse_args()

# Extract data and remove outliers:
fname = args.lc
t,f = np.loadtxt(fname,unpack=True,usecols=(0,1))
mflux = np.median(f)
f = f/mflux

ofname = args.lc_to_detrend
if ofname != '':
    tt,ff = np.loadtxt(ofname,unpack=True,usecols=(0,1))
    ff = ff/mflux

n_live_points = 1000

# Prepare the celerite term:
import celerite
from celerite import terms

class RotationTerm(terms.Term):
    parameter_names = ("log_amp", "log_timescale", "log_period", "log_factor")

    def get_real_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        f = np.exp(log_factor)
        return (
            np.exp(log_amp) * (1.0 + f) / (2.0 + f), 
            np.exp(-log_timescale),
        )   

    def get_complex_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        f = np.exp(log_factor)
        return (
            np.exp(log_amp) / (2.0 + f), 
            0.0,
            np.exp(-log_timescale),
            2*np.pi*np.exp(-log_period),
        )   

rot_kernel = terms.TermSum(RotationTerm(
    log_amp=np.log(np.var((f-np.median(f)))),
    log_timescale=np.log(10.0),
    log_period=np.log(3.0),
    log_factor=np.log(1.0)))

# Jitter term:
kernel_jitter = terms.JitterTerm(np.log(100*1e-6))

# Wrap GP object to compute likelihood
kernel = rot_kernel + kernel_jitter
gp = celerite.GP(kernel, mean=0.0)
gp.compute(t)
min_timescale = np.log(np.min(np.abs(np.diff(t)))/2.)
max_timescale = np.log(np.max(t)-np.min(t))

print('Setting maximum and minimum timescales of period to:',np.exp(min_timescale),np.exp(max_timescale))
# Now define MultiNest priors and log-likelihood:
def prior(cube, ndim, nparams):
    # Prior on "median flux" is uniform:
    cube[0] = utils.transform_uniform(cube[0],0.5,1.5)
    # Prior on log-"amplitude" is uniform:
    cube[1] = utils.transform_uniform(cube[1],-30,30.)
    # Prior on log-"timescale" is also uniform:
    cube[2] = utils.transform_uniform(cube[2],-30,30)
    # Prior on log-period:
    cube[3] = utils.transform_uniform(cube[3],min_timescale,max_timescale)
    # Prior on the log-factor:
    cube[4] = utils.transform_uniform(cube[4],-30.,30.)
    # Pior on the log-jitter term:
    cube[5] = utils.transform_uniform(cube[5],-100.,30.)

def loglike(cube, ndim, nparams):
    # Extract parameters:
    mflux,lA,lt,lP,lf,ljitter = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5]
    # Residuals of the model:
    residuals = f - mflux
    # Set GP parameter vector:
    gp.set_parameter_vector(np.array([lA,lt,lP,lf,ljitter]))
    # Evaluate log-likelihood:
    return gp.log_likelihood(residuals)

n_params = 6
out_file = 'out_multinest_trend_'
fout = fname.split('.')[0]
print(fout)
if not os.path.exists(fout):
    os.mkdir(fout)

import pickle
# If not ran already, run MultiNest, save posterior samples and evidences to pickle file:
if not os.path.exists(fout+'/posteriors_trend.pkl'):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    posterior_samples = output.get_equal_weighted_posterior()[:,:-1]
    a_lnZ = output.get_stats()['global evidence']
    logZ = (a_lnZ / np.log(10))
    out = {}
    out['posterior_samples'] = {}
    out['posterior_samples']['unnamed'] = posterior_samples
    out['posterior_samples']['mflux'] = posterior_samples[:,0]
    out['posterior_samples']['logB'] = posterior_samples[:,1]
    out['posterior_samples']['logL'] = posterior_samples[:,2]
    out['posterior_samples']['logP'] = posterior_samples[:,3]
    out['posterior_samples']['logC'] = posterior_samples[:,4]
    out['posterior_samples']['logjitter'] = posterior_samples[:,5]
    out['logZ'] = (a_lnZ / np.log(10))
    pickle.dump(out,open(fout+'/posteriors_trend.pkl','wb'))
else:
    out = pickle.load(open(fout+'/posteriors_trend.pkl','rb'))
    posterior_samples = out['posterior_samples']['unnamed']
    
#idx_large_P = np.where(np.exp(out['posterior_samples']['logP'])>8.)[0]
#print posterior_samples.shape
#posterior_samples = posterior_samples[idx_large_P,:]
#print posterior_samples.shape
# Extract posterior parameter vector:
theta = np.median(posterior_samples,axis=0)
one_array = np.ones(len(t))
ferr = one_array*np.exp(theta[-1])
gp.set_parameter_vector(theta[1:])

# Get prediction from GP:
x = np.linspace(np.min(t)-0.1, np.max(t)+0.1, 5000)
print('Getting prediction...')
pred_mean, pred_var = gp.predict((f-theta[0]), x, return_var=True)
pred_std = np.sqrt(pred_var)

print('Detrending...')
# Detrend extra light curve using the best-fit parameters, if given:
if ofname != '':
    opred_mean, opred_var = gp.predict((f-theta[0]), tt, return_var=True)
    print('f-theta:',f[:10]-theta[0])
    print('opred_mean:',opred_mean[:10])
    opred_std = np.sqrt(opred_var)
    print('ff:',ff[:10])
    ff = ff/(opred_mean+theta[0])
    print('ff/opred+theta',ff[:10])
    ff_err = np.sqrt(opred_var + np.median(ferr)**2)
    det_lc = open(fout+'/det_lc.dat','w')
    for i in range(len(tt)):
        det_lc.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(tt[i],ff[i],np.median(np.exp(out['posterior_samples']['logjitter']))))
    det_lc.close()

print('Plotting...')
# Plot:
sns.set_context("talk")
sns.set_style("ticks")
sns.set_context("talk")
sns.set_style("ticks")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['axes.linewidth'] = 1.2 
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['lines.markeredgewidth'] = 1 
fig = plt.figure(figsize=(10,5))
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3,0.8])

tzero = int(t[0])

# Plot solution:
period = np.median(np.exp(out['posterior_samples']['logP']))
sigma_period = np.sqrt(np.var(np.exp(out['posterior_samples']['logP'])))
jitter = np.median(np.exp(out['posterior_samples']['logjitter']))*1e6 # In ppm
ax = plt.subplot(gs[0])
color = "cornflowerblue"
plt.plot(t-tzero, f,".k",markersize=2,alpha=1.0,label='Data')
plt.plot(x-tzero, pred_mean + theta[0], linewidth=1, color=color,label='GP',alpha=1.0)
plt.fill_between(x-tzero,pred_mean + theta[0] - 1.*pred_std,pred_mean + theta[0] + 1.*pred_std, color=color,alpha=0.2)
plt.fill_between(x-tzero,pred_mean + theta[0] - 2.*pred_std,pred_mean + theta[0] + 2.*pred_std, color=color,alpha=0.2)
plt.fill_between(x-tzero,pred_mean + theta[0] - 3.*pred_std,pred_mean + theta[0] + 3.*pred_std, color=color,alpha=0.2)

plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
#plt.xlim(100,150)
plt.ylabel('Relative flux')
plt.legend(loc='upper right')
#plt.title('Best-fit period: ${0:.2f}\pm{1:.2f}$ days | Jitter: ${2:.2f}$ ppm'.format(period,sigma_period,jitter))
ax.xaxis.set_major_formatter(plt.NullFormatter())

# Get prediction from GP to get residuals:
ax = plt.subplot(gs[1])
pred_mean, pred_var = gp.predict((f-theta[0]), t, return_var=True)
plt.errorbar(t-tzero,(f-theta[0]-pred_mean)*1e6,yerr=np.ones(len(t))*jitter,fmt='.k',markersize=1,elinewidth=1,alpha=0.1)
time_res = t 
res = (f-theta[0]-pred_mean)*1e6
plt.xlabel('Time (BJD-'+str(tzero)+')')
plt.ylabel('Residuals')
plt.xlim(np.min(t-tzero)-0.1,np.max(t-tzero)+0.1)
#plt.xlim(100,150)
plt.ylim(-np.median(ferr*1e6)*5,np.median(ferr*1e6)*5)
plt.tight_layout()
plt.savefig(fout+'/GP_fit.pdf')
plt.savefig(fout+'/GP_fit.eps')

fig = plt.figure(figsize=(8,8))
plt.hist(np.exp(out['posterior_samples']['logP']),bins=30,normed=True)
plt.xlabel('Period (days)')
plt.ylabel('Posterior density')
plt.savefig(fout+'/Period_Posterior.pdf')
