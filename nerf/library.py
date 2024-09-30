import os
import pkgutil
import inspect
from types import ModuleType
from flytekit.core.python_function_task import PythonFunctionTask
from nerf import tasks  # Ensure this is an absolute import

class TaskLibrary:
    def __init__(self, module: ModuleType):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.visited_modules = set()
        self.task_names = {}
        self.tasks = self.search(module)
        
    def is_local_module(self, module):
        try:
            module_path = os.path.abspath(module.__file__)
            return module_path.startswith(self.project_root)
        except AttributeError:
            return False

    def search(self, obj):
        out = []
        
        if isinstance(obj, PythonFunctionTask):
            task_name = obj.name
            if task_name in self.task_names:
                raise ValueError(f"Duplicate task name found: {task_name}")
            self.task_names[task_name] = obj
            out.append(obj)
        elif isinstance(obj, ModuleType) and self.is_local_module(obj):
            if obj in self.visited_modules:
                return out
            self.visited_modules.add(obj)
            # Search for objects in the current module
            for name, member in inspect.getmembers(obj):
                out.extend(self.search(member))
            
            # Search for objects in submodules
            if hasattr(obj, '__path__'):
                for importer, modname, ispkg in pkgutil.iter_modules(obj.__path__):
                    submodule = importer.find_module(modname).load_module(modname)
                    if self.is_local_module(submodule):
                        out.extend(self.search(submodule))
                
        return out

# Example usage:
if __name__ == "__main__":
    library = TaskLibrary(tasks)
    print(task_library.tasks)