#include "../include/nnet.hpp"


float randFloat ()
{
	return ((float) rand()) / RAND_MAX;
}



void nnet::neural::regenUID ()
{

	m_UID = "";

	for (int i = 0; i < 8; ++i)
	{

		int num = rand() % 36;

		if (num < 10)
		{
			m_UID += std::to_string(num);
		}
		else
		{
			m_UID += ('a' + (num - 10));
		}

	}

}
