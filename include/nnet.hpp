#ifndef NNET_HPP
#define NNET_HPP

#include <memory>
#include <vector>
#include <string>

#include "nnet_error.hpp"


namespace nnet
{

	float randFloat ();



	struct layer;
	class population;

	class neural
	{

		public:
			neural (int middleLayerCount, int inputNodeCount, int middleNodeCount, int outputNodeCount);

			std::string getUID ();

			// use verbose makeCopy method to copy a neural network
			std::unique_ptr<neural> makeCopy ();

			std::vector<std::shared_ptr<layer>> layers;

			std::shared_ptr<layer> inputLayer;
			std::shared_ptr<layer> outputLayer;

			void calculate ();
			void randomize ();
			void tweak (float magnitude);

			void zeroInput ();


			// either of these selectOutput() functions should be run AFTER calculate()

			// this function uses the output values as probability weights to select one of the nodes
			// because the nodes can have negative values, this function adds 1.1 to each of the weights so that the range will be 0.1 to 2.1
			int selectOutput ();

			// this function simply returns the index of the greatst-valued output node
			// no randomness involved
			int selectOutputFixed ();


		// UID
		private:
			std::string m_UID;
			void regenUID ();


		// data properties
		private:
			int m_middleLayerCount;

			int m_inputNodeCount;
			int m_middleNodeCount;
			int m_outputNodeCount;


		// these are to be used only by the public makeCopy method
		private:
			neural (const neural&) = default;
			neural& operator = (const neural&) = default;

	};



	struct node
	{
		node (std::weak_ptr<layer> _prevLayer);

		std::weak_ptr<layer> prevLayer; 

		std::vector<float> weights;
		float bias = 0;

		void calculate ();
		void randomize ();
		void tweak (float magnitude);

		float value = 0;

	};



	struct layer
	{
		layer (int nodeCount, std::weak_ptr<layer> prevLayer = std::weak_ptr<layer>());

		std::vector<node> nodes;

		void calculate ();
		void randomize ();
		void tweak (float magnitude);

	};

}


#endif
