class ParamGroupMixin:
    """
    This mixin adds the param_group property which is the ordered list of parameters
    associated with a sampler. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._param_group = list()

    @property
    def param_group(self):
        return self._param_group

    @param_group.setter
    def param_group(self, value):
        self._param_group = value

    def __str__(self):
        param_names = [param.name for param in self.param_group]
        return '{} {}'.format(self.__class__.__name__, param_names)
