import abc

class BaseModel(abc.ABC):

	@abc.abstractmethod
	def run(self, *args, **kwargs):
		pass
