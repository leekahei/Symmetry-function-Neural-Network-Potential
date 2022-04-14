#!/usr/bin/env python
# coding: utf-8

import os
import re
import io as pio
import fnmatch
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import tables
import ase
from ase import io as ase_io
from ase import db as ase_db
from ase.db import core as db_core
from ase.calculators import calculator as ase_calc

def update_dataframe_from_geometries(df: pd.DataFrame,
                                     scalar_keys: List[str] = (),
                                     array_keys: List[str] = (),
                                     atoms_key: str = 'geometry',
                                     size_key: str = 'size',
                                     inplace: bool = True
                                     ) -> pd.DataFrame:
    """Intermediate function for object-dataframe consistency"""
    if not inplace:
        df = df.copy()
    geometries = df[atoms_key]
    scalar_idxs = []
    array_idxs = []
    for scalar in scalar_keys:
        if scalar not in df.columns:
            df[scalar] = pd.Series(dtype=float)
        scalar_idxs.append(df.columns.get_loc(scalar))
    if size_key not in df.columns:
        df[size_key] = pd.Series(dtype=int)
    size_idx = df.columns.get_loc(size_key)
    for array in array_keys:
        if array not in df.columns:
            df[array] = pd.Series(dtype=object)
        array_idxs.append(df.columns.get_loc(array))
    for idx, geom in enumerate(geometries):
        df.iat[idx, size_idx] = len(geom)
        for scalar, scalar_idx in zip(scalar_keys, scalar_idxs):
            try:
                df.iat[idx, scalar_idx] = geom.info[scalar]
            except KeyError:
                continue
        for array, array_idx in zip(array_keys, array_idxs):
            try:
                df.iat[idx, array_idx] = geom.arrays[array]
            except KeyError:
                continue
    return df

def update_geometries_from_calc(geometries: List[ase.Atoms],
                                energy_key: str = 'energy',
                                force_key: str = 'force'
                                ) -> List[ase.Atoms]:
    """Query attached calculators for energy and forces."""
    for idx, geom in enumerate(geometries):
        try:
            geom.info[energy_key] = geom.calc.get_potential_energy()
        except (ase_calc.PropertyNotImplementedError,
                AttributeError):
            pass  # no energy
        try:
            forces = geom.calc.get_forces()
        except (ase_calc.PropertyNotImplementedError,
                AttributeError):
            if force_key in geom.arrays:
                forces = geom.arrays[force_key]
            else:
                continue  # no forces
        try:
            geom.new_array('fx', forces[:, 0])
            geom.new_array('fy', forces[:, 1])
            geom.new_array('fz', forces[:, 2])
        except ValueError:  # shape mismatch
            continue
        except RuntimeError:  # array already exists
            continue
    return geometries

def parse_trajectory(fname: str,
                     scalar_keys: List[str] = (),
                     array_keys: List[str] = (),
                     prefix: str = None,
                     atoms_key: str = "geometry",
                     energy_key: str = "energy",
                     force_key: str = 'force',
                     size_key: str = 'size'):
    """
    Wrapper for ase.io.read, which is compatible with
    many file formats (notably VASP's vasprun.xml and extended xyz).
    If available, force information is written to each ase.Atoms object's
    arrays attribute as separate "fx", "fy", and "fz" entries.
    Args:
        fname (str): filename.
        scalar_keys (list): list of ase.Atoms.info keys to query and
            include as a DataFrame column. e.g. ["config_type"].
        array_keys (list): list of ase.Atoms.arrays keys to query and
            include as a DataFrame column. e.g. ["charge"].
        prefix (str): prefix for DataFrame index.
            e.g. "bulk" -> [bulk_0, bulk_1, bulk_2, ...]
        atoms_key (str): column name for geometries, default "geometry".
            Modify when parsed geometries are part of a larger pipeline.
        energy_key (str): column name for energies, default "energy".
        force_key (str): identifier for forces, default "force".
        size_key (str):  column name for number of atoms per geometry,
            default "size".
    Returns:
        df (pandas.DataFrame): standard dataframe with columns
           [atoms_key, energy_key, fx, fy, fz]
    """
    geometries = ase_io.read(fname, index=slice(None, None))
    new_index = None
    
    if not isinstance(geometries, list):
        geometries = [geometries]
    geometries = update_geometries_from_calc(geometries,
                                             energy_key=energy_key,
                                             force_key=force_key)
    # create DataFrame
    default_columns = [atoms_key, energy_key, 'fx', 'fy', 'fz']
    scalar_keys = [p for p in scalar_keys
                   if p not in default_columns]
    array_keys = [p for p in array_keys
                  if p not in default_columns]
    columns = default_columns + scalar_keys + array_keys
    df = pd.DataFrame(columns=columns)
    df[atoms_key] = geometries
    df[energy_key] = 0.0
    # object-dataframe consistency
    scalar_keys = scalar_keys + [energy_key]
    array_keys = array_keys + ["fx", "fy", "fz"]
    df = update_dataframe_from_geometries(df,
                                          atoms_key=atoms_key,
                                          size_key=size_key,
                                          scalar_keys=scalar_keys,
                                          array_keys=array_keys,
                                          inplace=True)
    if new_index is not None:
        df.index = new_index
        print('Loaded index from file:', fname)
    elif prefix is not None:
        pattern = '{}_{{}}'.format(prefix)
        df = df.rename(pattern.format)
    return df

class DataReader:    
    def __init__(self,
                 atoms_key = 'geometry',
                 energy_key = 'energy',
                 force_key = 'force',
                 size_key = 'size',
                 config_key = 'config_type',
                 overwrite=False
                 ):       
        """
        Args:
            atoms_key (str): column name for geometries, default "geometry".
                Modify when parsed geometries are part of a larger pipeline.
                
            energy_key (str): column name for energies, default "energy".
            
            force_key (str): identifier for forces, default "force".
            
            size_key (str): column name for number of atoms per geometry,
                default "size".
            
            config_key (str): column name for configuration type, default "config_type".
            
            overwrite (bool): Allow overwriting of existing DataFrame
                with matching key when loading.
        """        
        self.atoms_key = atoms_key
        self.energy_key = energy_key
        self.force_key = force_key
        self.size_key = size_key
        self.config_key = config_key
        self.overwrite = overwrite

        self.data = {}
        self.keys = []
        
    def __repr__(self):
        summary = ["DataReader:"]        
        if len(self.keys) == 0:
            summary.append(f"    Datasets: None")        
        else:
            summary.append(f"    Datasets: {len(self.keys)} ({self.keys})")        
        return "\n".join(summary)

    def __str__(self):
        return self.__repr__()
    
    def consolidate(self, remove_duplicates=True, keep='first'):
        dataframes = [self.data[k] for k in self.keys]
        df = pd.concat(dataframes)
        duplicate_array = df.index.duplicated(keep=keep)
        if np.any(duplicate_array):
            print('Duplicates keys found:', np.sum(duplicate_array))
            if remove_duplicates:
                print('Removing with keep=', keep)
                df = df[~duplicate_array]
                print('Unique keys:', len(df))
        return df
    
    def load_dataframe(self, dataframe, prefix=None):
        """Load existing pd.DataFrame"""
        for key in [self.atoms_key, self.energy_key, self.size_key]:
            if key not in dataframe.columns:
                raise RuntimeError("Missing \"{}\" column.".format(key))
        name_0 = dataframe.index[0]  # existing prefix takes priority
        if isinstance(name_0, str):
            if '_' in name_0:
                prefix = '_'.join(name_0.split('_')[:-1])
        if prefix is None:  # no prefix provided
            prefix = len(self.data)
            pattern = '{}_{{}}'.format(prefix)
            dataframe = dataframe.rename(pattern.format)
        if prefix in self.data:
            print('Data already exists with prefix "{}".'.format(prefix),
                  end=' ')
            if self.overwrite is True:
                print('Overwriting...')
                self.data[prefix] = dataframe
            else:
                print('Skipping...')
                return
        else:
            self.data[prefix] = dataframe
            self.keys.append(prefix)
    
    def dataframe_from_trajectory(self,
                                  filename,
                                  prefix=None,
                                  load=True,
                                  energy_key=None,
                                  force_key=None,
                                  size_key=None,
                                  config_key=None,
                                  **kwargs):
        """Wrapper for io.parse_trajectory()"""
        if prefix is None:
            prefix = len(self.data)
        if energy_key is None:
            energy_key = self.energy_key
        if force_key is None:
            force_key = self.force_key
        if size_key is None:
            size_key = self.size_key
        if config_key is None:
            config_key = self.config_key
        
        df = parse_trajectory(filename,
                              prefix = prefix, 
                              scalar_keys = [config_key],
                              atoms_key = self.atoms_key,
                              energy_key = energy_key,
                              force_key = force_key,
                              size_key = size_key,
                              **kwargs)

        if energy_key != self.energy_key:
            df.rename(columns={energy_key: self.energy_key},
                      inplace=True)
        if load:
            self.load_dataframe(df, prefix=prefix)
        else:
            return df

    dataframe_from_xyz = dataframe_from_trajectory
    dataframe_from_vasprun = dataframe_from_trajectory
