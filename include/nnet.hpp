#ifndef NNET_HPP
#define NNET_HPP

#include <memory>
#include <vector>
#include <string>

#include "nnet_error.hpp"


namespace nnet
{

	// wrapper for the builtin rand() function
	float randFloat ();

	// get endianness of current system
	bool isLittleEndian ();



	struct layer;
	class population;

	class neural
	{

		public:
			neural (int middleLayerCount, int inputNodeCount, int middleNodeCount, int outputNodeCount);

			// every neural object automatically creates its own unique UID
			// call this function to retrieve it
			std::string getUID ();



			// use verbose makeCopy method to copy a neural network
			// returns a pointer to the copy, which is allocated with the "new" keyword
			neural* makeCopy ();

			// functions to save and load to/from a file
			bool saveToFile (std::string filename);
			// this returns a pointer to an object allocated with the "new" keyword
			static neural* loadFromFile (std::string filename);



			std::vector<std::shared_ptr<layer>> layers;

			// pointers to input and output layers for convenience
			std::shared_ptr<layer> inputLayer;
			std::shared_ptr<layer> outputLayer;

			// forward calculation. make sure all inputs are set as desired before calling this
			void calculate ();
			// set all weights and biases to random values
			void randomize ();
			// tweak all weights/biases by random values, with a maximum magnitude parameter
			void tweak (float magnitude);


			// backprop, given an ideal output
			// set "accumulate" to true to average over a minibatch, then call backpropApply()
			void backprop (bool accumulate, float learningRate, std::vector<float> ideal);
			void backpropApply ();

			// merge backprop accumulation from another neural object
			void backpropMergeFrom (neural& other);
			// same as makeCopy, but point to same underlying weight data
			neural* split ();

			// clear the backprop accumulation data without applying it
			void backpropClear ();

			// backprop function uses this variable to keep track of how many datasets are in a batch
			int trainDataCount = 0;

			// get current cost value
			float cost (std::vector<float> ideal);



			// set all input nodes to value
			void clearInput (float value);


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


		private:
			// these are to be used only by the public makeCopy method
			neural* makeCopy_m (bool copyWeights);
			neural (const neural&) = default;
			neural& operator= (const neural&) = default;

			// move methods aren't needed right now, but I'll implement them if necessary
			neural (neural&&) = delete;
			neural& operator= (neural&&) = delete;

	};



	struct node
	{
		node (std::weak_ptr<layer> _prevLayer);

		// pointer to previous layer, so that this node can calculate what its value should be
		std::weak_ptr<layer> prevLayer; 

		std::shared_ptr<std::vector<float>> weights;
		float bias = 0;

		// see the descriptions in class neural{} for what these functions do
		void calculate ();
		void randomize ();
		void tweak (float magnitude);

		// the active value being held by this node
		float value = 0;


		////// backprop calculation
		// "ideal" argument is only used if "isOutputNode" is true
		void backprop (bool accumulate, float learningRate, bool isOutputNode, float ideal = 0);


		// call this after processing a minibatch, to actually apply the nudges
		void backpropApply (int trainDataCount);

		// clear the backprop accumulation data without applying it
		void backpropClear ();


		void inline activate ();
		float cost (float ideal);
		// this function should ONLY be called by the output layer nodes!!!
		float inline dCost_dValue (float ideal);
		// derivative of activation function
		float inline dValue_dUnactivated ();
		// indices denote which weight to calculate for
		float inline dUnactivated_dWeight (std::shared_ptr<layer> prevLayer, int weightInd);
		float constexpr dUnactivated_dBias ();
		float inline dUnactivated_dPrevValue (int prevNodeInd);

		// resets dCost_dValue_ of this node
		void resetVitalCache ();


		////// backprop cache data
		float dCost_dValue_ = 0; // this value is affected from outside this node

		////// backprop nudge sums (for minibatch averaging)
		std::vector<float> weightNudgeSums;
		float biasNudgeSum = 0;

	};



	struct layer
	{
		layer (int nodeCount, std::weak_ptr<layer> prevLayer = std::weak_ptr<layer>());

		std::vector<node> nodes;

		// these just call the respective functions on each of the nodes in this layer
		void calculate ();
		void randomize ();
		void tweak (float magnitude);

		// use this for the output layer
		void backprop (bool accumulate, float learningRate, std::vector<float> ideal);
		// use this for all middle layers
		void backprop (bool accumulate, float learningRate);

		// call this after processing a minibatch, to actually apply the nudges
		void backpropApply (int trainDataCount);

		// clear the backprop accumulation data without applying it
		void backpropClear ();

		void resetVitalCache ();

	};

}


#endif
