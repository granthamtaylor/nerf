from faker import Faker

from nerf.orchestration.constants import context
from nerf.core.structs import Hyperparameters

@context['basic'](cache=False)
def label(params: Hyperparameters) -> str:
    """Generate a label for the model"""
    
    Faker.seed(hash(tuple(vars(params).values())))
    
    fake = Faker()
    
    return fake.street_name().replace(" ", "-").lower()
    