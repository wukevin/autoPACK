#include as follow : execfile('pathto/POP37.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP37= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP37.sph',
radii = [[4.1500000000000004, 5.1799999999999997, 5.2999999999999998, 4.9100000000000001]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP37.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP37',
positions = [[(-0.53000000000000003, 1.79, 13.289999999999999), (-0.19, -1.97, 20.800000000000001), (1.3500000000000001, -0.17999999999999999, 3.9500000000000002), (2.8799999999999999, -2.6499999999999999, 12.720000000000001)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP37)
