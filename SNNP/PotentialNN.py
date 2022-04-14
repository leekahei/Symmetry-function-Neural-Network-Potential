#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import pi, cos, exp

import matplotlib.pyplot as plt

import pandas as pd

import ase
from ase.neighborlist import neighbor_list

from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

def cutoff_fc(dist, rmin, rmax):
    dist = np.asarray(dist)
    fc = np.ones_like(dist)
    mask = dist > rmin
    fc[mask] = 0.5*(cos((dist[mask]-rmin)*pi/(rmax-rmin)) + 1.0)
    fc[dist >= rmax] = 0.
    return fc

# G1
def G1(dist, rmin, rmax):
    g1 = np.sum(cutoff_fc(dist,rmin,rmax))
    return g1

# G2
def G2(dist, rmin, rmax, Rs, eta):
    fc = cutoff_fc(dist,rmin,rmax)
    dist = np.asarray(dist)
    dr = dist - Rs
    g2 = np.sum(exp(-eta*dr*dr)*fc)
    return g2

# G3
def G3(dist, rmin, rmax, kappa):
    fc = cutoff_fc(dist,rmin,rmax)
    dist = np.asarray(dist)
    g3 = np.sum(cos(dist*kappa)*fc)
    return g3

def structure_summary(dataframe, config_key = "config_type", size_key = "size"):
    data = dataframe
    d = list(dict.fromkeys(data[config_key]))

    df = pd.DataFrame({config_key:[], "data_size":[], size_key: []})

    L = []
    S = []
    for c in d:
        s = data[data[config_key]==c]
        s = s.iloc[:,s.columns.get_loc(size_key)].to_numpy()
        L.append(len(s))
        S.append(np.unique(s))
    
    df[config_key] = d
    df["data_size"] = L
    df[size_key] = S
        
    return df

def appending_zeros(dictionary):
    element_len = []
    
    for i, element in enumerate(dictionary):
        element_len.append(len(element))
    max_len = int(max(element_len))
    
    for i, element in enumerate(dictionary):
        e = np.zeros(max_len)
        e[:len(element)] = element
        dictionary[i] = e
    
    return dictionary
    
class featurization:
    def __init__(self,
                 config_key = "config_type",
                 geometry_key = "geometry",
                 size_key = "size", 
                 force_key = ["fx", "fy", "fz"],
                 energy_key = "energy"):
        """
        config_key (str): column name for configuration type, default "config_type".
        
        geometry_key (str): column name for geometry, default "geometry".
        
        size_key (str): column name for number of atoms per geometry, default "size".
        
        force_key (str): column name for forces, default ["fx", "fy", "fz"].
        """
        
        self.geometry_key = geometry_key
        self.config_key = config_key
        self.size_key = size_key
        self.force_key = force_key
        self.energy_key = energy_key
        
        self.data = {}
        self.summary = {}
        self.env = []
        self.geom = {}
        self.config = []
        self.size = []
        self.force = []
        self.energy = []
    
    def data_from_dataframe(self, data):
        data = data.copy()
        self.data = data
        self.summary = structure_summary(data, self.config_key)
        
        ## change positions to pair distance
        geo = self.data.iloc[:,self.data.columns.get_loc(self.geometry_key)]
        self.geom = geo
        pair_dist = geo.copy()
        for i, geom in enumerate(pair_dist):
            dist = np.trim_zeros(np.sort(np.ravel(geom.get_all_distances())))
            dist = dist[::2]
            pair_dist[i] = dist
        
        pair_dist = appending_zeros(pair_dist)
        
        self.env = pair_dist
        
        ## config
        self.config = self.data.iloc[:,self.data.columns.get_loc(self.config_key)].to_numpy()
        for i, c in enumerate(self.config):
            a = self.summary[self.summary[self.config_key] == c].index
            self.config[i] = a[0]
        
        ## size
        self.size = self.data.iloc[:,self.data.columns.get_loc(self.size_key)].to_numpy()
        
        ## calculate the force on each atom by f = sqrt(fx^2+fy^2+fz^2)
        self.force = self.data.iloc[:,self.data.columns.get_loc(self.force_key[0])].to_numpy()
        
        fx = self.data.iloc[:,self.data.columns.get_loc(self.force_key[0])].to_numpy()
        fy = self.data.iloc[:,self.data.columns.get_loc(self.force_key[1])].to_numpy()
        fz = self.data.iloc[:,self.data.columns.get_loc(self.force_key[2])].to_numpy()
            
        fsquare = fx**2+fy**2+fz**2

        for i, F in enumerate(fsquare):
            self.force[i] = np.sqrt(F)
        
        ## energy
        self.energy = self.data.iloc[:,self.data.columns.get_loc(self.energy_key)].to_numpy()
    
    def input_descriptor(self, symmetry_para = True,
                         rmin = 2.6, rmax = 6, Rs = [1.9], eta = [100], kappa = [8.5], cutoff = 3,
                         force = True,
                         config_type = True,
                         size = False):
        
        self.input_dimension = 0
        
        if symmetry_para:
            self.rmin = rmin
            self.rmax = rmax
            self.Rs = Rs
            self.eta = eta
            self.kappa = kappa
            self.cutoff = cutoff
        
        ## update the env to be symmetry and set as intially X if it is on
        if symmetry_para:
            
            atoms = self.data.iloc[:,self.data.columns.get_loc('geometry')].to_numpy()
            for index, atom in enumerate(atoms):
                G = []
                i, j, r_ij = neighbor_list('ijd', atom, self.cutoff, self_interaction=False)
                for a in range(max(i)+1):
                    g1 = G1(r_ij[i==a], self.rmin, self.rmax)
                    tmp = [g1]
                    for R in self.Rs:
                        for Eta in self.eta:
                            g2 = G2(r_ij[i==a], self.rmin, self.rmax, R, Eta)
                            tmp.append(g2)
                    
                    for K in kappa:
                        g3 = G3(r_ij[i==a], self.rmin, self.rmax, K)
                        tmp.append(g3)
                        
                    G.append(tmp)
                atoms[index] = np.array(G)
            
            self.env = atoms
            info = self.env
            self.input_dimension = len(self.env[0][0])
        
        ## if not using symmetry function, the pair dist is used
        else:
            info = self.env
            self.input_dimension = len(self.env[0])
        
        ## append force if it is on
        if force:
            if symmetry_para:
                for i, f in enumerate(self.force):
                    F = f.reshape(-1,1)
                    self.env[i] = np.concatenate((self.env[i], F), axis = 1)
            
                info = self.env
                self.input_dimension = len(self.env[0][0])
            else:
                forces = appending_zeros(self.force)
                for i, e in enumerate(self.env):
                    self.env[i] = np.append(e, forces[i])
                    
                info = self.env
                self.input_dimension = len(self.env[0])
        ## append config_type if it is on
        if config_type:
            if symmetry_para:
                for i, c in enumerate(self.config):
                    C = np.ones((self.env[i].shape[0], 1))*c
                    self.env[i] = np.concatenate((C, self.env[i]), axis = 1)
            
                info = self.env
                self.input_dimension = len(self.env[0][0])
            else:
                for i, e in enumerate(self.env):
                    self.env[i] = np.append(e, self.config[i])
                
                info = self.env
                self.input_dimension = len(self.env[0])
            
        ## append size if it is on
        if size:
            if symmetry_para:
                for i, s in enumerate(self.size):
                    S = np.ones((self.env[i].shape[0], 1))*s
                    self.env[i] = np.concatenate((S, self.env[i]), axis = 1)
            
                info = self.env
                self.input_dimension = len(self.env[0][0])
            else:
                for i, e in enumerate(self.env):
                    self.env[i] = np.append(e, self.size[i])
                
                info = self.env
                self.input_dimension = len(self.env[0])
            
        X = info
        
        if symmetry_para:
            copy = self.env.copy()
            for i, e in enumerate(self.energy):
                copy[i] = np.ones((self.env[i].shape[0], 1))*self.energy[i]/self.size[i]
                
            self.energy = copy
            
            Y = self.energy
            
        else:
            Y = self.energy
        
        return X, Y

class neural_network:
    def __init__(self, train_size = 0.2, layer_sizes = (10,10), random_state = 0, total_energy = True, symmetry_para = True):
        
        self.train_size = train_size
        self.test_size = 1 - train_size
        self.layer_sizes = layer_sizes
        self.random_state = random_state

        self.nn = MLPRegressor(hidden_layer_sizes=self.layer_sizes,
                                activation='logistic',
                                max_iter=6000,solver='lbfgs',
                                random_state = self.random_state)
        
        self.rs = ShuffleSplit(n_splits=1, test_size = self.test_size, train_size = self.train_size, random_state = self.random_state)
        self.train_index = []
        self.test_index = []
        
        if symmetry_para != True:
            total_energy = False
        
        self.total_energy = total_energy
        self.symmetry_para = symmetry_para
        
    def split(self,
                X, Y, filename = 'train_{}.txt'):
        self.X = X
        self.Y = Y
        
        if self.symmetry_para:
            x = np.concatenate(X, axis=0)
            y = np.concatenate(Y, axis=0)
            y = y.flatten()
        else:
            x = X.to_numpy()
            x = np.stack(x).astype(None)
            y = Y

        self.x = x
        self.y = y
        
        train, test = list(self.rs.split(self.y))[0]
        self.train_index = train
        self.test_index = test
        
        if filename:
            if '{}' in filename:
                filename = filename.format(len(train))
                
            if '.txt' in filename != True:
                filename = filename + '.txt'
            
            with open(filename, 'w') as f:
                f.writelines(["%s\n" % item  for item in train])
                
    def fit(self, filename = None):
        self.nn.fit(self.x[self.train_index],self.y[self.train_index])
        
        if filename:
            with open(filename, 'wb') as f:
                pickle.dump(self.nn, f)
        
    def load(self, filename):
        with open(filename) as f:
            self.nn = pickle.load(f)
            
    def predict(self, Print = True, Plot = True):
        y_predict = self.nn.predict(self.x)
        mse_test = mean_squared_error(self.y[self.test_index],y_predict[self.test_index])
        mse_train = mean_squared_error(self.y[self.train_index],y_predict[self.train_index])

        mae_test = mean_absolute_error(self.y[self.test_index],y_predict[self.test_index])
        mae_train = mean_absolute_error(self.y[self.train_index],y_predict[self.train_index])

        R2 = r2_score(self.y[self.test_index],y_predict[self.test_index])
        
        if Print:
            if self.symmetry_para:
                print('Prediction of energy per atom: \n')
            else:
                print('Prediction of total energy: \n')
            print(f'R2 score = {R2:8.5f}')
            print()
            print(f'MSE test  = {mse_test:8.5f}     MAE test  = {mae_test:8.5f}')
            print()
            print(f'MSE train = {mse_train:8.5f}     MAE train = {mae_train:8.5f}')
            print()
            
        if Plot:
            if self.symmetry_para:
                plt.figure()
                plt.plot(self.y, y_predict, 'x')
                plt.plot(np.arange(self.y.min(), self.y.max(), 0.1),
                         np.arange(self.y.min(), self.y.max(), 0.1), label = 'y=x')
                plt.xlabel('Actual energy per atom')
                plt.ylabel('Predicted energy per atom')
                plt.legend()
                plt.show()
            else:
                plt.figure()
                plt.plot(self.y, y_predict, 'x')
                plt.plot(np.arange(self.y.min(), self.y.max(), 0.1),
                         np.arange(self.y.min(), self.y.max(), 0.1), label = 'y=x')
                plt.xlabel('Actual total energy')
                plt.ylabel('Predicted total energy')
                plt.legend()
                plt.show()
        
        errors_energy_per_atom = [R2, mse_test, mse_train, mae_test, mae_train]
        
        
        if self.total_energy:
            predict_energy_per_atom = y_predict
            predict_energy_total = np.array([])
            for i, atoms in enumerate(self.X):
                l = int(len(atoms))
                e = np.sum(predict_energy_per_atom[:l])
                predict_energy_per_atom = np.delete(predict_energy_per_atom, np.arange(l))
                predict_energy_total = np.append(predict_energy_total, e)
            
            actual_energy_per_atom = self.y
            actual_energy_total = np.array([])
            for i, atoms in enumerate(self.Y):
                l = int(len(atoms))
                e = np.sum(actual_energy_per_atom[:l])
                actual_energy_per_atom = np.delete(actual_energy_per_atom, np.arange(l))
                actual_energy_total = np.append(actual_energy_total, e)
                
            
            train, test = list(self.rs.split(actual_energy_total))[0]
            
            mse_test = mean_squared_error(actual_energy_total[test],predict_energy_total[test])
            mse_train = mean_squared_error(actual_energy_total[train],predict_energy_total[train])

            mae_test = mean_absolute_error(actual_energy_total[test],predict_energy_total[test])
            mae_train = mean_absolute_error(actual_energy_total[train],predict_energy_total[train])

            R2 = r2_score(actual_energy_total[test],predict_energy_total[test])
            
            if Print:
                print('Prediction of total energy: \n')
                print(f'R2 score = {R2:8.5f}')
                print()
                print(f'MSE test  = {mse_test:8.5f}     MAE test  = {mae_test:8.5f}')
                print()
                print(f'MSE train = {mse_train:8.5f}     MAE train = {mae_train:8.5f}')
                print()
                
            if Plot:
                plt.figure()
                plt.plot(actual_energy_total, predict_energy_total, 'x')
                plt.plot(np.arange(min(actual_energy_total), max(actual_energy_total)),
                            np.arange(min(actual_energy_total), max(actual_energy_total)), label = 'y=x')
                plt.xlabel('Actual energy')
                plt.ylabel('Predicted energy')
                plt.legend()
                plt.show()
            
            errors_total_energy = [R2, mse_test, mse_train, mae_test, mae_train]
        
            return errors_energy_per_atom, errors_total_energy
        
        else:
            return errors_energy_per_atom
