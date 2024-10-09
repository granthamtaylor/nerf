import flytekit as fk
from faker import Faker

from nerf.orchestration.images import image
from nerf.core.structs import Hyperparameters

@fk.task(container_image=image)
def label(params: Hyperparameters) -> str:
    
    Faker.seed(hash(tuple(vars(params).values())))
    
    fake = Faker()
    
    return fake.street_name().replace(" ", "-").lower()
    