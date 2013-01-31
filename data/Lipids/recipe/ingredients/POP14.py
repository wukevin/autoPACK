#include as follow : execfile('pathto/POP14.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP14= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP14.sph',
radii = [[3.9700000000000002, 5.4299999999999997, 5.1699999999999999, 6.2800000000000002]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP14.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP14',
positions = [[(-1.98, -2.1699999999999999, 21.379999999999999), (1.0800000000000001, -0.31, 15.630000000000001), (2.7599999999999998, 3.6899999999999999, 4.8200000000000003), (1.72, -6.9699999999999998, 8.1799999999999997)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP14)