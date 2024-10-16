from typing_extensions import ParamSpec
import copy
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar
from functools import partial, wraps, cached_property

import flytekit
from union.actor import ActorEnvironment as UnionActor
from rich.panel import Panel
from rich.console import Console
from rich.pretty import Pretty


P = ParamSpec("P")
T = TypeVar("T")

# basically, I want the docstring for `flyte.task` to be available for users to see
# this is "copying" the docstring from `flyte.task` to functions wrapped by `forge`
# more details here: https://github.com/python/typing/issues/270
def forge(
    source: Callable[Concatenate[Any, P], T]
) -> Callable[[Callable], Callable[Concatenate[Any, P], T]]:
    def wrapper(target: Callable) -> Callable[Concatenate[Any, P], T]:
        @wraps(source)
        def wrapped(self, *args: P.args, **kwargs: P.kwargs) -> T:
            return target(self, *args, **kwargs)

        return wrapped

    return wrapper

def inherit(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    
    old = copy.deepcopy(old)
    new = copy.deepcopy(new)
    
    for key, value in new.items():
        if key in old:
            if isinstance(value, dict):
                old[key] = inherit(old[key], value)
            else:
                old[key] = value
        else:
            old[key] = value

    return old

class ActorEnvironment:
    
    @forge(UnionActor.__init__)
    def __init__(self, name: str, **overrides: Any) -> Any:
        
        self.name = name

        _overrides: dict[str, Any] = {"name": name}
        for key, value in overrides.items():

            _overrides[key] = value

        self.overrides = _overrides
        self.cached = False
    
    @cached_property
    def actor(self) -> UnionActor:
        self.cached = True
        return UnionActor(**self.overrides)

    @forge(UnionActor.__init__)
    def __call__(self, _task_function: Callable) -> Callable:
        
        return self.actor.task(_task_function)

    def show(self) -> None:
        
        console = Console()
        
        console.print(Panel.fit(Pretty(self.overrides), title=f"Actor {self.name}"))

    task = __call__

class TaskEnvironment:

    @forge(flytekit.task)
    def __init__(self, name: str, **overrides: Any) -> Any:
        
        self.name = name

        _overrides: dict[str, Any] = {}
        for key, value in overrides.items():

            if key == '_task_function':
                raise KeyError("Cannot override task function")

            _overrides[key] = value

        self.overrides = _overrides

    @forge(flytekit.task)
    def update(self, **overrides: Any) -> None:
        self.overrides = inherit(self.overrides, overrides)
        
    @forge(flytekit.task)
    def extend(self, name: str, **overrides: Any) -> "TaskEnvironment":
        return self.__class__(name=name, **inherit(self.overrides, overrides))
    
    @forge(flytekit.task)
    def __call__(self, _task_function: Callable|None=None, /, **overrides) -> Callable:        

        # no additional overrides are passed
        if _task_function is not None:
            
            if callable(_task_function):
                
                return partial(flytekit.task, **self.overrides)(_task_function)
        
            else:
                raise ValueError('The first positional argument must be a callable')
    
        # additional overrides are passed
        else:
            def inner(_task_function: Callable) -> Callable:

                inherited = inherit(self.overrides, overrides)
                
                return partial(flytekit.task, **inherited)(_task_function)
        
            return inner
    
    def show(self) -> None:
        
        console = Console()
        
        console.print(Panel.fit(Pretty(self.overrides), title=f"Task {self.name}"))

    task = __call__

class EnvironmentContext:

    def __init__(self, *environments: TaskEnvironment|ActorEnvironment) -> None:
        
        self.environments: dict[str, TaskEnvironment|ActorEnvironment] = {}
        
        for environment in environments:
            self.environments[environment.name] = environment
    
    def __getitem__(self, key: str) -> TaskEnvironment|ActorEnvironment:
        
        if key not in self.environments:
            raise KeyError(f"Environment {key} not found")
        
        return self.environments[key]

    def get(self, key: str) -> TaskEnvironment|ActorEnvironment:

        return self[key]

    def __setitem__(self, key: str, environment: TaskEnvironment|ActorEnvironment) -> None:
        
        if not isinstance(environment, TaskEnvironment|ActorEnvironment):
            raise TypeError("Value must be an instance of Environment")
        
        self.environments[key] = environment

    def add(self, environment: TaskEnvironment|ActorEnvironment) -> None:
        
        self[environment.name] = environment

    def __repr__(self) -> str:
        
        out = ""
        
        for value in self.environments.values():
            out += f"- {value.overrides}\n"
        return out

    def show(self) -> None:
        
        for environment in self.environments.values():
            environment.show()