"""Generates data for train/test algorithms"""
from datetime import datetime
import pickle
import os, sys, time
import random
import pandas as pd
import tldextract

from dga_family import banjori, corebot, cryptolocker, dircrypt, kraken, lockyv2, pykspa, qakbot, ramdo, ramnit, simda

# Our ourput file containg all the training data
DATA_FILE = sys.path[0]+'/data/traindata.pkl'

'''
Func: Read the Alexa top 1 million website file
'''
def get_alexa(num, filename=sys.path[0]+'/data/top-1m.csv'):
    f = open(filename, 'r')
    return [tldextract.extract(x.split(',')[1]).domain for x in f.readlines()[:num]]
"""
Func: Generates num_per_dga of each DGA
Params:
num_per_dga: each kind of dag generate %d fake data.
"""
def gen_malicious(num_per_dga=10000):
    domains = []
    labels = []
    # We use some arbitrary seeds to create domains with banjori
    banjori_seeds = ['somestring', 'firetruck', 'bulldozer', 'airplane', 'racecar',
                     'apartment', 'laptop', 'laptopcomp', 'malwareisbad', 'crazytrain',
                     'thepolice', 'fivemonkeys', 'hockey', 'football', 'baseball',
                     'basketball', 'trackandfield', 'fieldhockey', 'softball', 'redferrari',
                     'blackcheverolet', 'yellowelcamino', 'blueporsche', 'redfordf150',
                     'purplebmw330i', 'subarulegacy', 'hondacivic', 'toyotaprius',
                     'sidewalk', 'pavement', 'stopsign', 'trafficlight', 'turnlane',
                     'passinglane', 'trafficjam', 'airport', 'runway', 'baggageclaim',
                     'passengerjet', 'delta1008', 'american765', 'united8765', 'southwest3456',
                     'albuquerque', 'sanfrancisco', 'sandiego', 'losangeles', 'newyork',
                     'atlanta', 'portland', 'seattle', 'washingtondc']

    segs_size = max(1, int(num_per_dga/len(banjori_seeds)))
    for banjori_seed in banjori_seeds:
        domains += banjori.generate_domains(segs_size, banjori_seed)
        labels += ['banjori']*segs_size
    domains += corebot.generate_domains(num_per_dga)
    labels += ['corebot']*num_per_dga

    # Create different length domains using cryptolocker
    crypto_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(crypto_lengths)))
    for crypto_length in crypto_lengths:
        domains += cryptolocker.generate_domains(segs_size,
                                                 seed_num=random.randint(1, 1000000),
                                                 length=crypto_length)
        labels += ['cryptolocker']*segs_size

    domains += dircrypt.generate_domains(num_per_dga)
    labels += ['dircrypt']*num_per_dga

    # generate kraken and divide between configs
    kraken_to_gen = max(1, int(num_per_dga/2))
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'a', 3)
    labels += ['kraken']*kraken_to_gen
    domains += kraken.generate_domains(kraken_to_gen, datetime(2016, 1, 1), 'b', 3)
    labels += ['kraken']*kraken_to_gen

    # generate locky and divide between configs
    locky_gen = max(1, int(num_per_dga/11))
    for i in range(1, 12):
        domains += lockyv2.generate_domains(locky_gen, config=i)
        labels += ['locky']*locky_gen

    # Generate pyskpa domains
    domains += pykspa.generate_domains(num_per_dga, datetime(2016, 1, 1))
    labels += ['pykspa']*num_per_dga

    # Generate qakbot
    domains += qakbot.generate_domains(num_per_dga, tlds=[])
    labels += ['qakbot']*num_per_dga

    # ramdo divided over different lengths
    ramdo_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(ramdo_lengths)))
    for rammdo_length in ramdo_lengths:
        domains += ramdo.generate_domains(segs_size,
                                          seed_num=random.randint(1, 1000000),
                                          length=rammdo_length)
        labels += ['ramdo']*segs_size

    # ramnit
    domains += ramnit.generate_domains(num_per_dga, 0x123abc12)
    labels += ['ramnit']*num_per_dga

    # simda
    simda_lengths = range(8, 32)
    segs_size = max(1, int(num_per_dga/len(simda_lengths)))
    for simda_length in range(10, len(simda_lengths)+10):
        domains += simda.generate_domains(segs_size,
                                          length=simda_length,
                                          tld=None,
                                          base=random.randint(2, 2**32))
        labels += ['simda']*segs_size

    return domains, labels

"""
Func: Grab all data for train/test and save
Params:
force:If true overwrite, else skip if file already exists
"""
def gen_data(force=False):
    if force or (not os.path.isfile(DATA_FILE)):
        domains, labels = gen_malicious(10000)

        # Get equal number of benign/malicious
        mal_cnt = len(domains)
        domains += get_alexa(mal_cnt)
        labels += ['benign']*mal_cnt
        rand_idx = [random.uniform(0,1) for i in labels]
        new_df = pd.DataFrame({"label":labels, "text":domains, "rand":rand_idx})
        new_df = new_df.sort_values(by=["rand"])
        # Data Persistent Storage
        pickle.dump(new_df[["label", "text"]], open(DATA_FILE, 'wb'))

if __name__ == '__main__':
    start = time.time()
    gen_data(True)
    print (time.time()-start)
