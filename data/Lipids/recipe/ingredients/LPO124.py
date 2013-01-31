#include as follow : execfile('pathto/LPO124.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
LPO124= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/LPO124.sph',
radii = [[5.1299999999999999, 4.4500000000000002, 4.6600000000000001, 5.4800000000000004]],
cutoff_boundary = 0,
Type = 'MultiSphere',
cutoff_surface = 0,
gradient = '',
jitterMax = [0.5, 0.5, 0.10000000000000001],
packingPriority = 0,
rotAxis = [0.0, 2.0, 1.0],
nbJitter = 5,
molarity = 1.0,
rotRange = 6.2831,
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/LPO124.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'LPO124',
positions = [[(0.27000000000000002, 1.6799999999999999, -16.920000000000002), (0.14999999999999999, 3.6899999999999999, -6.8700000000000001), (-0.28000000000000003, -1.46, -24.550000000000001), (5.1100000000000003, 0.45000000000000001, -7.3600000000000003)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(LPO124)