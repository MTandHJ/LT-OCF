import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['Gowalla_m1', 'Yelp18_m1', 'AmazonBooks_m1', 'MovieLens1M_m2', 'AmazonCDs_m1', 'AmazonElectronics_m1']:
    dataset = dataloader.Loader(path="../../../data/" + world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print("adjoint method:", world.adjoint)
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'ltocf': model.LTOCF,
    'ltocf2': model.LTOCF2,
    'ltocf1': model.LTOCF1
}