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



void nnet::layer::backprop (bool accumulate, float learningRate, std::vector<float> ideal)
{

	for (int i = 0; i < nodes.size(); ++i)
	{
		node &n = nodes.at(i);
		n.backprop(accumulate, learningRate, true, ideal.at(i));
	}

}

void nnet::layer::backprop (bool accumulate, float learningRate)
{

	for (node &n: nodes)
	{
		n.backprop(accumulate, learningRate, false);
	}

}


void nnet::layer::backpropApply (int trainDataCount)
{

	for (node &n: nodes)
	{
		n.backpropApply(trainDataCount);
	}

}


void nnet::layer::resetVitalCache ()
{

	for (node &n: nodes)
	{
		n.resetVitalCache();
	}

}


