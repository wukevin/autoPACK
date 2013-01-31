#include as follow : execfile('pathto/POP82.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP82= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP82.sph',
radii = [[3.1400000000000001, 6.1799999999999997, 5.1399999999999997, 3.73]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP82.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP82',
positions = [[(-1.5800000000000001, 0.72999999999999998, -21.190000000000001), (5.0700000000000003, -3.1800000000000002, -2.5099999999999998), (4.4299999999999997, -2.2999999999999998, -9.9700000000000006), (0.029999999999999999, -0.37, -14.76)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP82)