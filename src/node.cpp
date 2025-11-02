#include "../include/nnet.hpp"

#include <cmath>

#include <iostream>

nnet::node::node (std::weak_ptr<layer> _prevLayer)
: prevLayer {_prevLayer}
{

	std::shared_ptr<layer> prevLayerLock = prevLayer.lock();

	if (prevLayerLock)
	{
		const int count = prevLayerLock->nodes.size();

		weights = std::make_shared<std::vector<float>>(count);
		//weights = std::vector<float>(count);
		weightNudgeSums = std::vector<float>(count, 0);

		if (!weights)
		{
			throw nnet::internalError("error initializing weight list, thrown from nnet::node::node()");
		}
	}

}



void nnet::node::calculate ()
{

	std::shared_ptr<layer> prevLayerLock = prevLayer.lock();
	if (!prevLayerLock)
	{
		throw nnet::internalError("could not access previous layer, thrown from nnet::node::calculate()");
	}
	if (prevLayerLock->nodes.size() != weights->size())
	{
		throw nnet::internalError("previous layer node count and this node's weight count do not match");
	}


	value = bias;

	for (int i = 0; i < weights->size(); ++i)
	{
		value += weights->at(i) * prevLayerLock->nodes.at(i).value;
	}

	activate();

}


void nnet::node::randomize ()
{

	bias = 2 * randFloat() - 1;

	for (int i = 0; i < weights->size(); ++i)
	{
		weights->at(i) = 2 * randFloat() - 1;
	}

}

void nnet::node::tweak (float magnitude)
{

	bias += magnitude * (2 * randFloat() - 1);

	for (int i = 0; i < weights->size(); ++i)
	{
		weights->at(i) += magnitude * (2 * randFloat() - 1);
	}

}




////// backprop functions

// make sure to call calculate() before this!
void nnet::node::backprop (bool accumulate, float learningRate, bool isOutputNode, float ideal)
{

	if (isOutputNode)
	{
		dCost_dValue(ideal);
	}
	//otherwise, the dCost_dValue_ for this node should already be set by the L+1 layer nodes


	// nudge the bias
	float dValue_dUnactivated_ = dValue_dUnactivated();
	float dCost_dBias = dCost_dValue_ * dValue_dUnactivated_ * dUnactivated_dBias();
	float delta = learningRate * dCost_dBias;
	if (accumulate)
	{
		biasNudgeSum -= delta;
	}
	else
	{
		bias -= delta;
	}


	std::shared_ptr<layer> prevLayerLock = prevLayer.lock();

	if (!prevLayerLock)
	{
		throw nnet::internalError("could not access previous layer, thrown from nnet::node::backprop()");
	}

	for (int i = 0; i < weights->size(); ++i)
	{
		// nudge the weights
		float dCost_dWeight = dCost_dValue_ * dValue_dUnactivated_ * dUnactivated_dWeight(prevLayerLock, i);
		float delta = learningRate * dCost_dWeight;
		if (accumulate)
		{
			weightNudgeSums.at(i) -= delta;
		}
		else
		{
			weights->at(i) -= delta;
		}


		// nudge the dCost_dValue of the L-1 layer nodes
		float dCost_dPrevValue = dCost_dValue_ * dValue_dUnactivated_ * dUnactivated_dPrevValue(i);
		// this has to be +=, not -= // also, this one is not scaled by learningRate
		prevLayerLock->nodes.at(i).dCost_dValue_ += dCost_dPrevValue;
	}


}



void nnet::node::backpropApply (int trainDataCount)
{

	bias += biasNudgeSum / trainDataCount;
	biasNudgeSum = 0;

	for (int i = 0; i < weights->size(); ++i)
	{
		weights->at(i) += weightNudgeSums.at(i) / trainDataCount;
		weightNudgeSums.at(i) = 0;
	}

}


void nnet::node::backpropClear ()
{
	biasNudgeSum = 0;

	for (float &f: weightNudgeSums)
	{
		f = 0;
	}
}



void inline nnet::node::activate ()
{
	// if this function is changed, remember to change the derivative: dValue_dUnactivated()
	value = tanh(value);
}

float nnet::node::cost (float ideal)
{
	return (value - ideal) * (value - ideal);
}

float inline nnet::node::dCost_dValue (float ideal)
{
	return dCost_dValue_ = 2 * (value - ideal);
}

// derivative of activation function
float inline nnet::node::dValue_dUnactivated ()
{
	return 1 / (cosh(value) * cosh(value));
}

// even though the node stores a weak_ptr to the previous layer, this takes a shared_ptr as an optimization
// so this function doesn't have to keep locking the weak_ptr every single time
float inline nnet::node::dUnactivated_dWeight (std::shared_ptr<nnet::layer> prevLayer, int weightInd)
{
	return prevLayer->nodes.at(weightInd).value;
}

float constexpr nnet::node::dUnactivated_dBias ()
{
	return 1;
}

float inline nnet::node::dUnactivated_dPrevValue (int prevNodeInd)
{
	return weights->at(prevNodeInd);
}


void nnet::node::resetVitalCache ()
{
	dCost_dValue_ = 0;
}


