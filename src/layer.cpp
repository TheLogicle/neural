#include "../include/nnet.hpp"

nnet::layer::layer (int nodeCount, std::weak_ptr<layer> prevLayer)
{

	nodes = std::vector<node>(nodeCount, {prevLayer});

}


void nnet::layer::calculate ()
{

	for (node &n: nodes)
	{
		n.calculate();
	}

}

void nnet::layer::randomize ()
{

	for (node &n: nodes)
	{
		n.randomize();
	}

}

void nnet::layer::tweak (float magnitude)
{

	for (node &n: nodes)
	{
		n.tweak(magnitude);
	}

}
