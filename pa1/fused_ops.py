from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        x = input_values[0] @ input_values[1]
        dims = tuple(range(-len(node.normalized_shape), 0))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var_eps = torch.var(x, dim=dims, unbiased=False, keepdim=True) + node.eps
        return (x - mean) / torch.sqrt(var_eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        """TODO: your code here"""
        # raise NotImplementedError
        A, B = node.inputs

        x = matmul(A, B)
        dims = tuple(range(-len(node.normalized_shape), 0))
        mu = mean(x, dim=dims, keepdim=True)
        x_centered = sub(x, mu)
        var = mean(mul(x_centered, x_centered), dim=dims, keepdim=True)
        sigma_squared = add_by_const(var, node.eps)
        sigma = sqrt(sigma_squared)
        mean_dy = mean(output_grad, dim=dims, keepdim=True)
        mean_dy_x_centered = mean(mul(output_grad, x_centered), dim=dims, keepdim=True)

        grad_ln = div(
            sub(
                sub(output_grad, mean_dy),
                mul(x_centered, div(mean_dy_x_centered, sigma_squared))
            ),
            sigma
        )
        grad_A = matmul(grad_ln, transpose(B, -1, -2)) 
        grad_B = matmul(transpose(A, -1, -2), grad_ln)
        return [grad_A, grad_B]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        """TODO: your code here"""
        x = input_values[0] @ input_values[1]
        x_max = torch.max(x, dim=-1, keepdim=True).values
        x_exp = torch.exp(x - x_max) 
        x_sum = x_exp.sum(dim=-1, keepdim=True)
        return x_exp / x_sum

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        """TODO: your code here"""
        A, B = node.inputs

        dim = node.dim
        s = node
        s_times_grad = output_grad * s
        sum_s_times_grad = sum_op(s_times_grad, dim=dim, keepdim=True)
        grad_softmax = s * sub(output_grad, sum_s_times_grad)

        grad_A = matmul(grad_softmax, transpose(B, -1, -2)) 
        grad_B = matmul(transpose(A, -1, -2), grad_softmax)
        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()