import os
import yaml
from munch import DefaultMunch

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_config_object(config_path) -> DefaultMunch:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return DefaultMunch.fromDict(config)


if __name__ == "__main__":
    config = create_config_object(config_path=os.path.join(project_path, "smplify3d/config/PW3D.yaml"))
    
    print(config.dataset.name)
