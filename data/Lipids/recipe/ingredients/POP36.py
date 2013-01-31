#include as follow : execfile('pathto/POP36.py',globals(),{'recipe':recipe_variable_name})
from AutoFill.Ingredient import SingleSphereIngr, MultiSphereIngr
POP36= MultiSphereIngr( 
packingMode = 'random',
color = [1, 0, 0],
sphereFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/recipe/ingredients/POP36.sph',
radii = [[6.5899999999999999, 4.4299999999999997, 6.0, 4.8499999999999996]],
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
meshFile = '/Users/ludo/DEV/autofill_googlesvn/data/Lipids/geoms/ingredients_1/POP36.c4d',
perturbAxisAmplitude = 0.1,
principalVector = [0.0, 0.0, 1.0],
name = 'POP36',
positions = [[(1.5700000000000001, -1.74, 21.100000000000001), (-5.2300000000000004, 9.6699999999999999, 9.2200000000000006), (-3.1200000000000001, -3.25, 10.07), (-1.1100000000000001, 5.6299999999999999, 17.66)]],
placeType = 'jitter',
useRotAxis = 1,
nbMol = 0,
)
recipe.addIngredient(POP36)