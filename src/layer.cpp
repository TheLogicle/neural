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



void nnet::layer::backprop (float learningRate, std::vector<float> ideal)
{

	for (int i = 0; i < nodes.size(); ++i)
	{
		node &n = nodes.at(i);
		n.backprop(learningRate, true, ideal.at(i));
	}

}

void nnet::layer::backprop (float learningRate)
{

	for (node &n: nodes)
	{
		n.backprop(learningRate, false);
	}

}


void nnet::layer::resetVitalCache ()
{

	for (node &n: nodes)
	{
		n.resetVitalCache();
	}

}


