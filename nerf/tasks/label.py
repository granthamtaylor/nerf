import flytekit
from faker import Faker

from nerf.orchestration.constants import image
from nerf.core.structs import Hyperparameters

@flytekit.task(container_image=image)
def label(params: Hyperparameters) -> str:
    """Generate a label for the model"""
    
    Faker.seed(hash(tuple(vars(params).values())))
    
    fake = Faker()
    
    return fake.street_name().replace(" ", "-").lower()
    