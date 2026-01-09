#include "../include/nnet.hpp"


nnet::neural::neural (int middleLayerCount, int inputNodeCount, int middleNodeCount, int outputNodeCount)
: m_middleLayerCount {middleLayerCount},
	m_inputNodeCount {inputNodeCount},
	m_middleNodeCount {middleNodeCount},
	m_outputNodeCount {outputNodeCount}
{

	if (middleLayerCount < 0)
	{
		throw nnet::usageError("Middle layer count must be >= 0");
	}
	if (inputNodeCount < 1)
	{
		throw nnet::usageError("Input node count must be >= 1");
	}
	if (middleNodeCount < 1)
	{
		throw nnet::usageError("Middle node count must be >= 1");
	}
	if (outputNodeCount < 1)
	{
		throw nnet::usageError("Output node count must be >= 1");
	}

	regenUID();

	std::shared_ptr<layer> inpLayer = std::make_shared<layer>(inputNodeCount);
	layers.emplace_back(inpLayer);

	inputLayer = inpLayer;

	for (int i = 0; i < middleLayerCount; ++i)
	{
		std::weak_ptr<layer> prevLayer = layers.back();
		std::shared_ptr<layer> middleLayer = std::make_shared<layer>(middleNodeCount, prevLayer);
		layers.emplace_back(middleLayer);
	}

	std::weak_ptr<layer> prevLayer = layers.back();
	std::shared_ptr<layer> outLayer = std::make_shared<layer>(outputNodeCount, prevLayer);
	layers.emplace_back(outLayer);

	outputLayer = outLayer;

}


std::string nnet::neural::getUID ()
{
	return m_UID;
}

nnet::neural* nnet::neural::makeCopy ()
{
	return makeCopy_m(true);
}

nnet::neural* nnet::neural::split ()
{
	return makeCopy_m(false);
}

nnet::neural* nnet::neural::makeCopy_m (bool copyWeights)
{

	neural* copiedNeural = new neural(*this);

	copiedNeural->regenUID();

	for (int i = 0; i < copiedNeural->layers.size(); ++i)
	{

		std::shared_ptr<layer> copiedLayer = std::make_shared<layer>( *(copiedNeural->layers.at(i)) );

		copiedNeural->layers.at(i) = copiedLayer;


		if (i == 0) continue;

		for (node &n: copiedLayer->nodes)
		{
			// fix the prevLayer pointers for all layers (except first layer)
			n.prevLayer = copiedNeural->layers.at(i - 1);

			// also copy the weights
			if (copyWeights) n.weights = std::make_shared<std::vector<float>>(*n.weights);
		}

	}

	copiedNeural->inputLayer = copiedNeural->layers.front();
	copiedNeural->outputLayer = copiedNeural->layers.back();

	copiedNeural->backpropClear();

	return copiedNeural;

}





void nnet::neural::calculate ()
{

	// start at index 1 because the input layer does not need to be calculated
	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->calculate();
	}

}


void nnet::neural::randomize ()
{

	// start at index 1 because the input layer does not need to be randomized
	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->randomize();
	}

}

void nnet::neural::tweak (float magnitude)
{

	// start at index 1 because the input layer does not need to be tweaked
	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->tweak(magnitude);
	}

}


void nnet::neural::backprop (bool accumulate, float learningRate, std::vector<float> ideal)
{

	// backprop cache doesn't need to be reset every time if processing a minibatch
	// also, update trainDataCount variable
	if (accumulate)
	{
		++trainDataCount;
	}
	else
	{
		trainDataCount = 0;
	}


	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->resetVitalCache();
	}


	outputLayer->backprop(accumulate, learningRate, ideal);

	// iterate backwards through all middle layers
	for (int i = layers.size() - 2; i >= 1; --i)
	{
		layers.at(i)->backprop(accumulate, learningRate);
	}

}


void nnet::neural::backpropApply ()
{

	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->backpropApply(trainDataCount);

		// technically, this operation isn't needed, but it's good for formality
		layers.at(i)->resetVitalCache();
	}

	trainDataCount = 0;

}


void nnet::neural::backpropClear ()
{
	for (int i = 1; i < layers.size(); ++i)
	{
		layers.at(i)->backpropClear();
	}

	trainDataCount = 0;
}



float nnet::neural::cost (std::vector<float> ideal)
{
	float costSum = 0;

	for (int i = 0; i < outputLayer->nodes.size(); ++i)
	{
		node &n = outputLayer->nodes.at(i);

		costSum += n.cost(ideal.at(i));
	}

	return costSum;
}



void nnet::neural::clearInput (float value)
{
	for (node &n: inputLayer->nodes)
		n.value = value;
}

void nnet::neural::setInput (const std::vector<float> &input)
{
	for (size_t i = 0; i < inputLayer->nodes.size(); ++i)
		inputLayer->nodes[i].value = input.at(i);
}



// this one simply selects the greatest valued output node as the selection
// no randomness involved
int nnet::neural::selectOutputFixed ()
{

	float max = -100;
	int maxInd = -1;

	for (int i = 0; i < outputLayer->nodes.size(); ++i)
	{
		node &n = outputLayer->nodes.at(i);

		if (n.value > max)
		{
			max = n.value;
			maxInd = i;
		}
	}

	if (maxInd == -1) throw nnet::internalError("invalid selection in neural::selectOutput");

	return maxInd;

}



// use the neural network's output as probability weights for each possible selection
int nnet::neural::selectOutput ()
{

	std::vector<float> weights;
	float weightSum = 0;

	// need to add 1.1 to the values because they could be negative
	// this converts the range into [0.1, 2.1]
	for (node &n: outputLayer->nodes)
	{
		weights.push_back(n.value + 1.1);
		weightSum += n.value + 1.1;
	}

	float randNum = randFloat() * weightSum;

	for (int i = 0; i < weights.size(); ++i)
	{
		if (weights.at(i) > randNum)
		{
			return i;
		}

		randNum -= weights.at(i);
	}

	// if control reaches here, randNum is likely almost equal to weightSum

	return weights.size() - 1;

}

