#include as follow : execfile('pathto/POP35.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP35= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP35.sph',
radii = [[5.6500000000000004, 3.7599999999999998, 6.0700000000000003, 4.6799999999999997]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP35.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP35',
positions = [[(1.5, 1.78, 21.469999999999999), (-4.5999999999999996, -4.4699999999999998, 6.9699999999999998), (1.47, 3.3799999999999999, 9.5399999999999991), (-3.0899999999999999, -4.3799999999999999, 16.329999999999998)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP35)