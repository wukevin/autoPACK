#include as follow : execfile('pathto/POP26.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP26= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP26.sph',
radii = [[6.5300000000000002, 6.1500000000000004, 2.8900000000000001, 5.9800000000000004]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP26.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP26',
positions = [[(-1.0700000000000001, 4.6699999999999999, -4.3399999999999999), (2.0899999999999999, -5.1500000000000004, -5.3099999999999996), (3.1200000000000001, 1.6699999999999999, -20.449999999999999), (0.34999999999999998, 0.050000000000000003, -15.85)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP26)
