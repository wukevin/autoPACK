#include as follow : execfile('pathto/POP10.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP10= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP10.sph',
radii = [[4.2000000000000002, 6.6399999999999997, 3.6600000000000001, 4.5899999999999999]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP10.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, -1.0],
name = 'POP10',
positions = [[(4.8200000000000003, -1.3300000000000001, -13.050000000000001), (0.54000000000000004, -7.3200000000000003, -9.7300000000000004), (-3.1899999999999999, -4.0, -14.4), (1.23, -0.34000000000000002, -21.0)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP10)
