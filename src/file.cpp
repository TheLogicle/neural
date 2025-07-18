#include "../include/nnet.hpp"

#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>


void compatCheck ()
{
	if (sizeof(float) != 4)
	{
		throw nnet::incompatibleError("Cannot save to file; incompatible machine.");
	}
}

// serialize a number into chars, then push it to vec at the BACK
template <typename T>
void serializePush (std::vector<char> &vec, T x, bool reverse)
{
	char* ptr = (char*) &x;

	if (reverse)
	{
		for (int i = sizeof(T) - 1; i >= 0; --i)
		{
			vec.push_back(ptr[i]);
		}
	}
	else
	{
		for (int i = 0; i < sizeof(T); ++i)
		{
			vec.push_back(ptr[i]);
		}
	}
}


// deserialize first few bytes of char vec into a number, then pop those chars from the FRONT of the vec
template <typename T>
T deserializePop (std::vector<char> &vec, bool reverse)
{

	if (reverse)
	{
		// reverse the bytes in-place
		for (int i = 0; i < sizeof(T) / 2; ++i)
		{
			char temp = vec.at(i);
			vec.at(i) = vec.at(sizeof(T) - 1 - i);
			vec.at(sizeof(T) - 1 - i) = temp;
		}
	}

	T x;
	std::memcpy(&x, vec.data(), sizeof(T));

	vec.erase(vec.begin(), vec.begin() + sizeof(T));

	return x;

}



bool nnet::neural::saveToFile (std::string filename)
{

	compatCheck();

	bool isBigEndian = !isLittleEndian();


	std::ofstream f1(filename, std::ios::binary);

	if (!f1) return false;


	std::vector<char> buf;


	serializePush<uint32_t>(buf, m_middleLayerCount, isBigEndian);
	serializePush<uint32_t>(buf, m_inputNodeCount, isBigEndian);
	serializePush<uint32_t>(buf, m_middleNodeCount, isBigEndian);
	serializePush<uint32_t>(buf, m_outputNodeCount, isBigEndian);

	f1.write(buf.data(), buf.size());


	// weights
	for (int i = 1; i < layers.size(); ++i)
	{
		std::shared_ptr<layer> l = layers.at(i);

		buf = std::vector<char>();

		for (int j = 0; j < l->nodes.size(); ++j)
		{

			node &n = l->nodes.at(j);

			for (int k = 0; k < n.weights.size(); ++k)
			{
				float weight = n.weights.at(k);

				serializePush<float>(buf, weight, isBigEndian);
			}
		}

		f1.write(buf.data(), buf.size());

	}



	// biases
	buf = std::vector<char>();

	for (int i = 1; i < layers.size(); ++i)
	{
		std::shared_ptr<layer> l = layers.at(i);

		for (int j = 0; j < l->nodes.size(); ++j)
		{
			float bias = l->nodes.at(j).bias;

			serializePush<float>(buf, bias, isBigEndian);
		}
	}

	f1.write(buf.data(), buf.size());

	f1.close();

	return true;

}

#include <iostream>

nnet::neural* nnet::neural::loadFromFile(std::string filename)
{

	compatCheck();

	bool isBigEndian = !isLittleEndian();


	std::ifstream f1(filename, std::ios::binary);

	if (!f1) return nullptr;

	std::vector<char> buf(4 * 4);

	f1.read(buf.data(), 4 * 4);


	int middleLayerCount = deserializePop<uint32_t>(buf, isBigEndian);
	int inputNodeCount = deserializePop<uint32_t>(buf, isBigEndian);
	int middleNodeCount = deserializePop<uint32_t>(buf, isBigEndian);
	int outputNodeCount = deserializePop<uint32_t>(buf, isBigEndian);

	neural* n1 = new neural(middleLayerCount, inputNodeCount, middleNodeCount, outputNodeCount);


	// weights
	for (int i = 1; i < n1->layers.size(); ++i)
	{
		std::shared_ptr<layer> l = n1->layers.at(i);
		std::shared_ptr<layer> prevLayer = n1->layers.at(i - 1);

		buf = std::vector<char>(4 * l->nodes.size() * prevLayer->nodes.size());

		f1.read(buf.data(), buf.size());

		for (int j = 0; j < l->nodes.size(); ++j)
		{
			node &n = l->nodes.at(j);

			for (int k = 0; k < n.weights.size(); ++k)
			{
				n.weights.at(k) = deserializePop<float>(buf, isBigEndian);
			}
		}
	}



	// biases
	buf = std::vector<char>(4 * (middleLayerCount * middleNodeCount + outputNodeCount));

	f1.read(buf.data(), buf.size());

	for (int i = 1; i < n1->layers.size(); ++i)
	{
		std::shared_ptr<layer> l = n1->layers.at(i);

		for (int j = 0; j < l->nodes.size(); ++j)
		{
			node &n = l->nodes.at(j);

			n.bias = deserializePop<float>(buf, isBigEndian);
		}
	}



	f1.close();

	return n1;

}


