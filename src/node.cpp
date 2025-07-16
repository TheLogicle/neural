#include "../include/nnet.hpp"

#include <cmath>

nnet::node::node (std::weak_ptr<layer> _prevLayer)
: prevLayer {_prevLayer}
{

	std::shared_ptr<layer> prevLayerLock = prevLayer.lock();

	if (prevLayerLock)
	{
		weights = std::vector<float>(prevLayerLock->nodes.size());
	}

}



void nnet::node::calculate ()
{

	std::shared_ptr<layer> prevLayerLock = prevLayer.lock();
	if (!prevLayerLock)
	{
		throw nnet::internalError("could not access previous layer");
	}
	if (prevLayerLock->nodes.size() != weights.size())
	{
		throw nnet::internalError("previous layer node count and this node's weight count do not match");
	}


	value = bias;

	for (int i = 0; i < weights.size(); ++i)
	{
		value += weights.at(i) * prevLayerLock->nodes.at(i).value;
	}

	value = tanh(0.7 * value);

}


void nnet::node::randomize ()
{

	bias = 2 * randFloat() - 1;

	for (int i = 0; i < weights.size(); ++i)	
	{
		weights.at(i) = 2 * randFloat() - 1;
	}

}

void nnet::node::tweak (float magnitude)
{

	bias += magnitude * (2 * randFloat() - 1);

	for (int i = 0; i < weights.size(); ++i)
	{
		weights.at(i) += magnitude * (2 * randFloat() - 1);
	}

}


