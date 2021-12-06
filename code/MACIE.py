import os
import sys
import getopt
import numpy as np
import copy
from itertools import product
from math import *
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

###
# FUNCTIONS
###

def hermite(points, z):
  p1 = 1/pi**0.4
  p2 = 0
  for j in range(points):
    p3 = p2
    p2 = p1
    p1 = z * np.sqrt(2/float(j+1)) * p2 - np.sqrt(j/float(j+1)) * p3
  
  pp = np.sqrt(2*points) * p2
  return p1, pp


'''
Given
  - points = number of fixed abscissae
  - iterlim = max number of Netwon-Rhapson iterations
Return gauss-hermite quadrature
  - points
  - weights
'''
def gauss_hermite(points, iterlim=50):
  x = np.array([0.0] * points)
  w = np.array([0.0] * points)
  for i in range(int(ceil(points/2))):
    if i==0:
      z = np.sqrt(2*points+1)-2*(2*points+1)**(-1.0/6)
    elif i == 1:
      z = z-np.sqrt(points)/z
    elif i == 2 or i == 3:
      z = 1.9*z - 0.9 * x[i-2]
    else:
      z = 2*z-x[i-2]
    for j in range(iterlim):
      z1 = z
      p0, p1 = hermite(points, z)
      z = z1- p0/p1
      if np.abs(z-z1) <= 1e-15:
        break
    
    x[i] = z
    x[points-1-i] = -z
    w[i] = 2/p1**2
    w[points-1-i] = 2/p1**2
  
  return x*np.sqrt(2), w/np.sum(w)


'''
Given
  - n_factor = number of factors
  - n_fix = number of fixed abscissae per dimension
  - prune = percentage of smallest weights to prune
Return n_factor-variate gauss-hermite quadrature
  - points
  - weights
'''
def mgauss_hermite(n_factor, n_fix=10, prune=20):
  gh = gauss_hermite(n_fix)
  pts = []
  wts = []
  for abscissa in product(range(n_fix),repeat=n_factor):
    pts.append([gh[0][i] for i in abscissa])
    wts.append(np.prod([gh[1][i] for i in abscissa]))
  
  if n_factor > 1 and prune > 0:
    threshold = np.percentile(wts,prune)
    pts = [pts[i] for i in range(n_fix**n_factor) if wts[i] > threshold]
    wts = [wts[i] for i in range(n_fix**n_factor) if wts[i] > threshold]
  return np.matrix(pts).T, np.array(wts)


'''
Given
  - a column of covariates
  - a column of outcomes
  - an array of weights
Return
  - column of coefficients from weighted linear regression
'''
def linear_regression(covariate,outcome,weights=None):
  N = len(outcome)
  X = np.concatenate((np.matrix([1]*N).T,covariate),axis=1)
  Y = outcome
  if weights is None:
    coef = np.linalg.inv(X.T * X) * X.T * Y
  else:
    W = np.array(weights)
    XtW = np.matrix(np.array(X.T) * W)
    coef = np.linalg.inv(XtW * X) * XtW * Y
  return coef


'''
Given
  - a column of covariates
  - a column of outcomes
  - maximum number of iterations for re-weighted least squares
Return
  - column of coefficients from logistic regression
'''
def logistic_regression(covariate,outcome,iterlim=50):
  N = len(outcome)
  X = np.concatenate((np.matrix([1]*N).T,covariate),axis=1)
  Y = outcome
  coef0 = linear_regression(covariate,outcome)
  converge = False
  i = 0
  while (converge == False and i < 50):
    eta = X * coef0
    mu = np.exp(eta)/(1+np.exp(eta))
    Z = eta + (Y-mu)/mu/(1-mu)
    weights = np.array(mu.T)*(1-np.array(mu.T))
    coef1 = linear_regression(covariate,Z,weights=weights)
    if (float(sum(abs(coef1-coef0))) < 1e-4):
      converge = True
    else:
      i = i + 1
      coef0 = coef1
  
  return coef1


'''
Given
  - command line arguments
Return
  - input file path
  - output folder path
'''
def readcommandline(argv):
  annotationfile = ''
  classfile = ''
  outputfolder = ''
  parameterfolder = ''
  factors = ''
  try:
    opts, args = getopt.getopt(argv,"m:a:c:o:f:p:",["mode=","afile=","cfile=","ofolder=","factors","pfolder="])
  except getopt.GetoptError:
    print 'usage: test.py -m <mode> -a <annotationfile> -c <classfile> -o <outputfolder> -f <factors> -p <parameterfolder>'
    sys.exit(2)
  for opt, arg in opts:
    if opt in ("-m", "--mode"):
      if arg in ("train","predict"):
        mode = arg
      else:
        print "<mode> options include 'train' and 'predict'"
        sys.exit(2)
    elif opt in ("-a", "--afile"):
      annotationfile = arg
    elif opt in ("-c", "--cfile"):
      classfile = arg
    elif opt in ("-o", "--ofolder"):
      outputfolder = arg
    elif opt in ("-f", "--factors"):
      factors = map(int,list(arg))
    elif opt in ("-p", "--pfolder"):
      parameterfolder = arg
  return mode, annotationfile, classfile, outputfolder, factors, parameterfolder


'''
Read file and find
  - 1st line: annotation names 
  - 2nd line: type ('D'==bernoulli or 'C'==normal)
  - annotation scores
Return
  - vector of annotation names
  - vector of annotation types
  - matrix of annotation scores
'''
def readfile(annotationfile, classfile):
  cfile = open(classfile,'r')
  line1 = cfile.readline().split()
  line2 = cfile.readline().split()
  line3 = map(int, cfile.readline().split())
  ML = len(line1)
  M = max(line3)
  
  afile = open(annotationfile,'r')
  afile.readline()
  annotation = []
  data_type = []
  data = [[] for j in range(M)]
  for line in afile:
    scores = line.split()
    for j in range(M):
      data[j].append([scores[x] for x in range(ML) if line3[x]==(j+1)])
  
  for j in range(M):
    annotation.append([line1[x] for x in range(ML) if line3[x]==(j+1)])
    data_type.append([line2[x] for x in range(ML) if line3[x]==(j+1)])
    data[j] = np.array(data[j],dtype="float")
  
  return annotation, data_type, data


'''
Given group j's
  - annotation scores
  - annotation types
  - number of factors
Return
  - initial parameters Zeta_j for the GLMM
'''
def find_Zeta_j_0(data_j, data_type_j, n_factors_j=1):
  # initialization
  dataX = np.matrix(data_j)
  L_j = len(data_type_j)
  beta_j_0 = [[0,0] for k in range(L_j)]
  
  # k-means clustering for functional prediction
  status_j_0 = KMeans(n_clusters=2).fit_predict(dataX)
  if (np.mean(status_j_0)>0.5):
    status_j_0 = 1-status_j_0
  covariate = np.matrix(status_j_0).T
  
  # linear or logistic regression for estimating beta coefficients
  for k in range(L_j):
    outcome = np.matrix(data_j[0:,k]).T
    if (data_type_j[k] == 'D'):
      coef = logistic_regression(covariate,outcome)
    elif (data_type_j[k] == 'C'):
      coef = linear_regression(covariate,outcome)
    beta_j_0[k] = [float(coef[0]), float(coef[1])]
  
  # factor analysis for estimating loading factors and uniquenesses
  model = FactorAnalysis(n_components=n_factors_j,max_iter=50).fit(dataX)
  Lambda_j_0 = model.components_
  phi_j_0 = model.noise_variance_
  
  # functional proportion
  gamma_j_0 = np.mean(status_j_0)
  
  return [np.array(beta_j_0), Lambda_j_0.T, phi_j_0, gamma_j_0]


'''
Given
  - annotation scores
  - annotation types
  - number of factors
Return
  - initial parameters for Zeta_j
  - initial parameters for posterior probabilities gamma (e.g., Pr(0,0), Pr(0,1), Pr(1,0), Pr(1,1))
'''
def find_Zeta_0(data, data_type, n_factors):
  M = len(data_type)
  Zeta_0 = []
  for j in range(M):
    Zeta_0.append(find_Zeta_j_0(data[j],data_type[j],n_factors[j]))
  
  gamma_0 = {}
  for c_i in product(range(2),repeat=M):
    gamma_0_c = 1
    for j in range(M):
      gamma_0_c *= Zeta_0[j][3]**c_i[j] * (1-Zeta_0[j][3])**(1-c_i[j])
    
    gamma_0[c_i]=gamma_0_c
  
  Zeta_0.append(gamma_0)
  return Zeta_0


'''
Given
- parameterfolder
- group j
Return
- all estimated parameters for Zeta_j except gamma_j
'''
def find_Zeta_j_hat(parameterfolder, j):
  # hat_betaphi
  beta_j_hat = []
  phi_j_hat = []
  file = open(parameterfolder+'hat_betaphi.txt','r')
  file.readline()
  for line in file:
    betaphi = line.split()
    if int(betaphi[1])-1==j:
        beta_j_hat.append([float(betaphi[3]),float(betaphi[4])])
        phi_j_hat.append(float(betaphi[5]))
  
  file.close()
  
  # hat_lambda
  Lambda_j_hat = []
  file = open(parameterfolder+'hat_lfactor_'+str(j+1)+'.txt','r')
  file.readline()
  for line in file:
    Lambda = line.split()
    lenLambda = len(Lambda)
    Lambda_j_hat.append([float(Lambda[2+k]) for k in range(lenLambda-2)])
  
  file.close()
  
  return [np.array(beta_j_hat), np.array(Lambda_j_hat), np.array(phi_j_hat)]


'''
Given
- parameterfolder
- annotation types
Return
- all estimated parameters for Zeta_j
- estimated parameters for posterior probabilities gamma
'''
def find_Zeta_hat(parameterfolder, data_type):
  M = len(data_type)
  gamma_j = [0 for x in range(M)]
  gamma_hat = {}
  file = open(parameterfolder+'hat_gamma.txt','r')
  for line in file:
    gamma = line.split()
    vector = eval(''.join(gamma[:M]))
    gamma_hat[vector] = float(gamma[M])
    for j in range(M):
        if vector[j]==1:
            gamma_j[j] = gamma_j[j]+float(gamma[M])
    
  
  file.close()
  
  Zeta_hat = []
  for j in range(M):
    Zeta_hat.append(find_Zeta_j_hat(parameterfolder, j)+[gamma_j[j]])
  
  Zeta_hat.append(gamma_hat)
  return Zeta_hat


'''
Compute random effect matrix (outcome x abscissae)
for each annotation group.
'''
def compute_b(Zeta_r, n_factors, quadratures):
  M = len(n_factors)
  b_jkt = []
  for j in range(M):
    b_jkt.append(Zeta_r[j][1]*quadratures[n_factors[j]][0])
  
  return b_jkt


'''
Using multivariate gauss-hermite quadratures, the expectation of
f(y_ij|c_ij,b_ij)f(b_ij) can be rewritten as a sum of weighted
f(y_ij|c_ij,b_ij) over all b_ij quadratures. This function
calculates each term in the sum, as well as the sum (integral).
'''
def compute_wfy(data, data_type, Zeta_r, quadratures, b_jkt):
  N = len(data[0])
  M = len(data)
  L = [len(data_type_j) for data_type_j in data_type]
  n_abscissae = [b_jkt[j].shape[1] for j in range(M)]
  
  # compute weighted f(y_ijk|c_ij,b_jkt)
  wfy, wfy_int = {}, {}
  for c_i in product(range(2),repeat=M):
    wfy[c_i] = [np.ones((N,n_abscissae[j])) for j in range(M)]
    wfy_int[c_i] = [[] for j in range(M)]
    for j in range(M):
      for k in range(L[j]):
        fixed = Zeta_r[j][0][k][0] + c_i[j]*Zeta_r[j][0][k][1]
        if (data_type[j][k]=='C'):
          eta = np.array((fixed + b_jkt[j][k,:]).T)
          y_ijk = np.tile(data[j][:,k],(n_abscissae[j],1))
          wfy[c_i][j] = wfy[c_i][j] * 1/sqrt(2*pi*Zeta_r[j][2][k]) * np.exp(-np.power(np.subtract(y_ijk,eta).T,2)/2/Zeta_r[j][2][k])
        else:
          eta = np.array(fixed + b_jkt[j][k,:])[0]
          y_ijk = np.array(np.matrix(data[j][:,k]).T)
          #print "the value of eta is:", eta
          #print "the value of y_ijk is:", y_ijk
          wfy[c_i][j] = wfy[c_i][j] * np.exp(y_ijk * eta) / (1+np.exp(eta))
      
    
  for c_i in product(range(2),repeat=M):
    for j in range(M):
      wfy[c_i][j] = wfy[c_i][j] * quadratures[n_factors[j]][1]
      wfy_int[c_i][j] = np.sum(wfy[c_i][j],axis=1)
    
  
  return wfy, wfy_int


'''
Compute the denominator in all posterior expectations.
'''
def compute_denominator(wfy_int, Zeta_r):
  N = len(wfy_int[wfy_int.keys()[0]][0])
  M = len(wfy_int[wfy_int.keys()[0]])
  denominator = np.zeros(N)
  for c_i in Zeta_r[M].keys():
    temp = np.ones(N)
    for j in range(M):
      temp = temp * wfy_int[c_i][j]
    
    denominator = denominator + temp * Zeta_r[M][c_i]
  
  print "denominator is:", denominator
  print "min value of denominator is:", np.min(denominator)
  print "max value of denominator is:", np.max(denominator)
  print "length of denominator is:", len(denominator)
  return denominator


'''
Compute the posterior probability of functional status
given annotation data.
'''
def compute_posterior(wfy_int, Zeta_r, denominator):
  N = len(wfy_int[wfy_int.keys()[0]][0])
  M = len(Zeta_r)-1
  posterior = {}
  for c_i in Zeta_r[M].keys():
    posterior[c_i] = np.ones(N)
    for j in range(M):
      posterior[c_i] = posterior[c_i] * wfy_int[c_i][j]
    
    posterior[c_i] = posterior[c_i] * Zeta_r[M][c_i] / denominator
  print "posterior is:", posterior
  return posterior


'''
Compute posterior expectations for the E-step.
'''
def compute_expectations(data, data_type, wfy, wfy_int, Zeta_r, n_factors, quadratures, b_jkt):
  N = len(data[0])
  M = len(data_type)
  L = [len(data_type[j]) for j in range(M)]
  exx, exe, ee2, eff, efe = {}, {}, {}, {}, {}
  for c_i in Zeta_r[M].keys():
    exx[c_i], exe[c_i], ee2[c_i], eff[c_i], efe[c_i] = [[] for j in range(M)], [[] for j in range(M)], [[] for j in range(M)], [[] for j in range(M)], [[] for j in range(M)]
    for j in range(M):
      exx[c_i][j], exe[c_i][j], ee2[c_i][j] = [np.zeros(N) for k in range(L[j])], [np.zeros(N) for k in range(L[j])], [np.zeros(N) for k in range(L[j])]
      eff[c_i][j], efe[c_i][j] = [[] for k in range(L[j])], [[] for k in range(L[j])]
      temp = np.prod([wfy_int[c_i][j2] for j2 in range(M) if j2 != j], axis=0)
      for k in range(L[j]):
        eff[c_i][j][k] = [[np.zeros(N) for s in range(n_factors[j])] for r in range(n_factors[j])]
        efe[c_i][j][k] = [np.zeros(N) for r in range(n_factors[j])]
        fixed = Zeta_r[j][0][k][0] + c_i[j]*Zeta_r[j][0][k][1]
        for t in range(wfy[c_i][j].shape[1]):
          f_jt = np.array(quadratures[n_factors[j]][0][:,t])
          eta = fixed + b_jkt[j][k,t]
          #print "the value of eta is:", eta
          prob = np.exp(eta)/(1+np.exp(eta))
          #print "the value of prob is:", prob
          if (data_type[j][k]=='C'):
            exx[c_i][j][k] = exx[c_i][j][k] + wfy[c_i][j][:,t]
            exe[c_i][j][k] = exe[c_i][j][k] + (data[j][:,k]-b_jkt[j][k,t]) * wfy[c_i][j][:,t]
            ee2[c_i][j][k] = ee2[c_i][j][k] + np.power(data[j][:,k]-eta,2) * wfy[c_i][j][:,t]
            for r in range(n_factors[j]):
              efe[c_i][j][k][r] = efe[c_i][j][k][r] + f_jt[r] * (data[j][:,k]-fixed) * wfy[c_i][j][:,t]
              for s in range(r,n_factors[j]):
                eff[c_i][j][k][r][s] = eff[c_i][j][k][r][s] + f_jt[r] * f_jt[s] * wfy[c_i][j][:,t]
              
            
          else:
            exx[c_i][j][k] = exx[c_i][j][k] + prob * (1-prob) * wfy[c_i][j][:,t]
            exe[c_i][j][k] = exe[c_i][j][k] + (data[j][:,k] - prob) * wfy[c_i][j][:,t]
            for r in range(n_factors[j]):
              efe[c_i][j][k][r] = efe[c_i][j][k][r] + f_jt[r] * (data[j][:,k]-prob) * wfy[c_i][j][:,t]
              for s in range(r,n_factors[j]):
                eff[c_i][j][k][r][s] = eff[c_i][j][k][r][s] + f_jt[r] * f_jt[s] * prob * (1-prob) * wfy[c_i][j][:,t]
              
            
        exx[c_i][j][k] = exx[c_i][j][k] * temp
        exe[c_i][j][k] = exe[c_i][j][k] * temp
        ee2[c_i][j][k] = ee2[c_i][j][k] * temp
        for r in range(n_factors[j]):
          efe[c_i][j][k][r] = efe[c_i][j][k][r] * temp
          for s in range(n_factors[j]):
            eff[c_i][j][k][r][s] = eff[c_i][j][k][r][s] * temp
          
        
      
    
  return exx, exe, ee2, eff, efe


'''
Single iteration of the EM algorithm.
Given previous parameter estimates Zeta_r, estimate new parameters Zeta_rp1.
'''
def EM(data, data_type, Zeta_r, n_factors, quadratures):
  N = len(data[0])
  M = len(data_type)
  L = [len(data_type[j]) for j in range(M)]
  
  b_jkt = compute_b(Zeta_r, n_factors, quadratures)
  
  # E-step
  wfy, wfy_int = compute_wfy(data, data_type, Zeta_r, quadratures, b_jkt)
  denominator = compute_denominator(wfy_int, Zeta_r)
  posterior = compute_posterior(wfy_int, Zeta_r, denominator)
  exx, exe, ee2, eff, efe = compute_expectations(data, data_type, wfy, wfy_int, Zeta_r, n_factors, quadratures, b_jkt)
  Zeta_rp1 = copy.deepcopy(Zeta_r)
  
  # M-step
  for j in range(M):
    for k in range(L[j]):
      exx_jk = np.matrix( np.sum( [Zeta_r[M][c_i] * np.matrix([[1,c_i[j]],[c_i[j],c_i[j]]]) * np.sum(exx[c_i][j][k]/denominator) for c_i in Zeta_r[M].keys()],axis=0) )
      exe_jk = np.matrix( np.sum( [Zeta_r[M][c_i] * np.matrix([[1],[c_i[j]]]) * np.sum(exe[c_i][j][k]/denominator) for c_i in Zeta_r[M].keys()],axis=0) )
      eff_jk = np.matrix( np.zeros((n_factors[j],n_factors[j])) )
      efe_jk = np.matrix( [[0] for r in range(n_factors[j])] )
      for r in range(n_factors[j]):
        efe_jk[r,0] = np.sum( [Zeta_r[M][c_i] * np.sum(efe[c_i][j][k][r] / denominator) for c_i in Zeta_r[M].keys()])
        for s in range(r,n_factors[j]):
          eff_jk[r,s] = np.sum( [Zeta_r[M][c_i] * np.sum(eff[c_i][j][k][r][s] / denominator) for c_i in Zeta_r[M].keys()])
          eff_jk[s,r] = eff_jk[r,s]
        
      
      beta_jk = np.array(np.linalg.inv(exx_jk)*exe_jk)
      Lambda_jk = np.linalg.inv(eff_jk)*efe_jk
      if (data_type[j][k]=='C'):
        phi_jk = np.sum([ Zeta_r[M][c_i] * ee2[c_i][j][k] / denominator for c_i in Zeta_r[M].keys()]) / N
        Zeta_rp1[j][0][k] = np.array(beta_jk.T)[0]
        Zeta_rp1[j][1][k] = np.array(Lambda_jk.T)[0]
        Zeta_rp1[j][2][k] = phi_jk
      else:
        phi_jk = -1
        Zeta_rp1[j][0][k][0] = Zeta_r[j][0][k][0]+beta_jk[0]
        Zeta_rp1[j][0][k][1] = Zeta_r[j][0][k][1]+beta_jk[1]
        Zeta_rp1[j][1][k] = Zeta_r[j][1][k] + np.array(Lambda_jk[0])
        Zeta_rp1[j][2][k] = phi_jk
    
    Zeta_rp1[j][3] = np.sum([np.mean(posterior[c_i]) for c_i in Zeta_r[M].keys() if c_i[j]==1])
  
  for c_i in Zeta_r[M].keys():
    Zeta_rp1[M][c_i] = np.mean(posterior[c_i])
  
  return Zeta_rp1, posterior


'''
Given Zeta, update parameter_r with Zeta intercepts, slopes, and loading factors.
'''
def update_parameter(parameter_r, Zeta_r, n_factors):
  M = len(Zeta_r)-1
  L = [len(Zeta_r[j][2]) for j in range(M)]
  parameter_r['intercept'].append([Zeta_r[j][0][k][0] for j in range(M) for k in range(L[j])])
  parameter_r['slope'].append([Zeta_r[j][0][k][1] for j in range(M) for k in range(L[j])])
  parameter_r['Lambda'].append([Zeta_r[j][1][k,f] for j in range(M) for k in range(L[j]) for f in range(n_factors[j])])
  
  return parameter_r


'''
Compute relative change in parameters between two Zeta's.
'''
def compute_diff(Zeta_1, Zeta_2):
  M = len(Zeta_1)-1
  diff = 0
  for j in range(M):
    diff = diff + np.sum(np.abs(Zeta_1[j][0]-Zeta_2[j][0]))
  total = 0
  for j in range(M):
    total = total + np.sum(np.abs(Zeta_1[j][0]))
  
  return diff * 1.0 / total


'''
Multi-dimensional Annotation Class Integrative Estimation (MACIE)
'''
def MACIE(data, data_type, Zeta_0, n_factors, criteria=1e-4, max_iter=100):
  
  quadratures = {}
  for n_factor in sorted(set(n_factors)):
    if (n_factor==1):
      prune = 0
    quadratures[n_factor] = mgauss_hermite(n_factor)
  
  parameter_r = {'intercept':[],'Lambda':[],'slope':[]}
  
  Zeta_r = copy.deepcopy(Zeta_0)
  parameter_r = update_parameter(parameter_r, Zeta_r, n_factors)
  Zeta_rp1, posterior = EM(data, data_type, Zeta_r, n_factors, quadratures)
  parameter_r = update_parameter(parameter_r, Zeta_rp1, n_factors)
  diff = compute_diff(Zeta_r,Zeta_rp1)
  iter = 2
  while (diff > 1e-4 and iter <= max_iter):
    print 'Iteration ' + str(iter-1) + ' complete, diff=' + str(diff)
    
    Zeta_r = copy.deepcopy(Zeta_rp1)
    Zeta_rp1, posterior = EM(data, data_type, Zeta_r, n_factors, quadratures)
    parameter_r = update_parameter(parameter_r, Zeta_rp1, n_factors)
    diff = compute_diff(Zeta_r,Zeta_rp1)
    iter = iter + 1
  
  print 'Iteration ' + str(iter-1) + ' complete, diff=' + str(diff)
  return Zeta_rp1, posterior, diff, iter-1, parameter_r


def save_Zeta(Zeta_r, n_factors, directory, annotation, prefix='', suffix=''):
  
  M = len(annotation)
  L = [len(annotation[j]) for j in range(M)]
  
  # save beta coefficients
  file = open(directory+prefix+'betaphi'+suffix+'.txt','w')
  file.write('annotation\tj\tk\tintercept\tslope\tphi\n')
  for j in range(M):
    for k in range(L[j]):
      file.write('\t'.join([annotation[j][k],str(j+1),str(k+1),str(Zeta_r[j][0][k][0]),str(Zeta_r[j][0][k][1]),str(Zeta_r[j][2][k])])+'\n')
    
  file.close()
  
  # save loading factors
  for j in range(M):
    file = open(directory+prefix+'lfactor_'+str(j+1)+suffix+'.txt','w')
    file.write('annotation\tk\t'+'\t'.join(['lfactor_'+str(j+1)+str(f+1) for f in range(n_factors[j])])+'\n')
    for k in range(L[j]):
      file.write('\t'.join([annotation[j][k],str(k+1)]+[str(Zeta_r[j][1][k,f]) for f in range(n_factors[j])])+'\n')
    
  file.close()
  
  
  # save posterior means
  file = open(directory+prefix+'gamma'+suffix+'.txt','w')
  for c_i in Zeta_r[M].keys():
    file.write(str(c_i)+'\t'+str(Zeta_r[M][c_i])+'\n')
  file.close()


def save_posterior(posterior, annotationfile, classfile, directory, prefix='', suffix=''):
  info = open(classfile,'r')
  info_name = info.readline().split()
  info_width = len(info_name)
  info_type = info.readline().split()
  info.close()
  
  snp_info = open(annotationfile,'r')
  snp_info.readline()
  file = open(directory+prefix+suffix+'.txt','w')
  file.write('\t'.join([info_name[x] for x in range(info_width) if info_type[x]=='I']+[str(c_i) for c_i in posterior.keys()])+'\n')
  i = 0
  for line in snp_info:
    temp = line.split()
    file.write('\t'.join([temp[x] for x in range(info_width) if info_type[x]=='I']+[str(posterior[c_i][i]) for c_i in posterior.keys()])+'\n')
    i = i+1
  
  snp_info.close()
  file.close()


def save_parameter(parameter_r, annotation, n_factors, directory, prefix='', suffix=''):
  M = len(annotation)
  L = [len(annotation[j]) for j in range(M)]
  
  # save intercepts
  file = open(directory+prefix+'intercept'+suffix+'.txt','w')
  file.write('\t'.join([annotation[j][k] for j in range(M) for k in range(L[j])])+'\n')
  for r in range(len(parameter_r['intercept'])):
    file.write('\t'.join([str(x) for x in parameter_r['intercept'][r]])+'\n')
  file.close()
  
  # save slopes
  file = open(directory+prefix+'slope'+suffix+'.txt','w')
  file.write('\t'.join([annotation[j][k] for j in range(M) for k in range(L[j])])+'\n')
  for r in range(len(parameter_r['slope'])):
    file.write('\t'.join([str(x) for x in parameter_r['slope'][r]])+'\n')
  file.close()
  
  # save loading factors
  file = open(directory+prefix+'lfactor'+suffix+'.txt','w')
  file.write('\t'.join([annotation[j][k] for j in range(M) for k in range(L[j]) for f in range(n_factors[j])])+'\n')
  file.write('\t'.join([str(f+1) for j in range(M) for k in range(L[j]) for f in range(n_factors[j]) ])+'\n')
  for r in range(len(parameter_r['Lambda'])):
    file.write('\t'.join([str(x) for x in parameter_r['Lambda'][r]])+'\n')
  file.close()

  
###
# MAIN
###

# An example of the coding_class.txt file for non-synonymous coding variants can be found here (https://github.com/xihaoli/MACIE/blob/main/code/coding_class.txt)
# The coding_annotations.txt file should have the same header as the coding_class.txt file
# python MACIE.py -m train -a train_coding_dir/coding_annotations.txt -c train_coding_dir/coding_class.txt -o train_coding_dir/ -f 22
# python MACIE.py -m predict -a pred_coding_dir/coding_annotations.txt -c train_coding_dir/coding_class.txt -o pred_coding_dir/ -f 22 -p train_coding_dir/

# An example of the noncoding_class.txt file for non-coding and synonymous coding variants can be found here (https://github.com/xihaoli/MACIE/blob/main/code/noncoding_class.txt)
# The noncoding_annotations.txt file should have the same header as the noncoding_class.txt file.
# python MACIE.py -m train -a train_noncoding_dir/noncoding_annotations.txt -c train_noncoding_dir/noncoding_class.txt -o train_noncoding_dir/ -f 23
# python MACIE.py -m predict -a pred_noncoding_dir/noncoding_annotations.txt -c train_noncoding_dir/noncoding_class.txt -o pred_noncoding_dir/ -f 23 -p train_noncoding_dir/

if __name__ == "__main__":
  mode, annotationfile, classfile, outputfolder, n_factors, parameterfolder = readcommandline(sys.argv[1:])
  annotation, data_type, data = readfile(annotationfile, classfile)
  if mode == "train":
    Zeta_0 = find_Zeta_0(data, data_type, n_factors)
    Zeta_hat, posterior, diff, iter, parameter_r = MACIE(data, data_type, Zeta_0, n_factors, criteria=1e-4, max_iter=200)
    save_Zeta(Zeta_hat, n_factors, outputfolder, annotation, prefix='hat_')
    save_parameter(parameter_r, annotation, n_factors, outputfolder, prefix='chain_')
  else:
    Zeta_hat = find_Zeta_hat(parameterfolder, data_type)
    Zeta_hatp1, posterior, diff, iter, parameter_r = MACIE(data, data_type, Zeta_hat, n_factors, criteria=1e-4, max_iter=1)
    save_posterior(posterior, annotationfile, classfile, outputfolder, prefix='MACIE')
