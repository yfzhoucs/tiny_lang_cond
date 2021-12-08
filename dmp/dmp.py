import torch.nn as nn


class DynamicMovementPrimitives(nn.Module):
	def __init__(self, num_basis_function):
		super(DynamicMovementPrimitives, self).__init__()

		self.num_basis_function = num_basis_function
		