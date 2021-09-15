import abc
import attr

class MSIBrain(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __len__(self):
        """Number of modalities."""
        
    @abc.abstractmethod
    def __getitem__(self, modality):
        """"""
        
    @abc.abstractmethod
    def response(self):
        """"""


@attr.s
class ErnstBanks2002(MSIBrain):
    
    mean = attr.ib(default=0., converter=float)
    std = attr.ib(default=1., converter=float) 
    weighs = attr.ib(default=(.5, .5))
    _result = attr.ib(factory=dict)
    
    def auditiva_modality(self, x):
        if "auditiva" in self._result:
            raise ValueError()
        ...
        self._result["auditiva"] = ...
        
    def visual_modality(self, x):
        if "visual" in self._result:
            raise ValueError()
        ...
        self._result["visual"] = ...
        
    def __len__(self):
        return 2
    
    def __getitem__(self, modality):
        if modality == "auditiva":
            return self.auditiva_modality
        elif modality == "visual":
            return self.visual_modality
        raise KeyError(modality)
        
    def response(self):
        return ...
    
    
ErnstBanks2002()