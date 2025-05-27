from pyModelChecking import *
from pyModelChecking.LTL import *

# Define atomic propositions as strings
user_consent = 'user_consent'
delete_content = 'delete_content'

# Define the LTL formula
phi = G(~user_consent >> ~delete_content)

# Define a simple Kripke structure
K = Kripke(
    R=[(0, 1), (1, 2), (2, 0)],
    L={
        0: {'user_consent'},
        1: set(),
        2: {'delete_content'}
    }
)

# Perform model checking
result = modelcheck(K, phi)

print(f"The formula {phi} is {'satisfied' if result else 'not satisfied'} in the model.")


