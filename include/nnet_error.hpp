// don't manually include this file!
// it is automatically included with nnet.hpp


#ifndef NNET_ERROR_HPP
#define NNET_ERROR_HPP

#include <stdexcept>

namespace nnet
{

	struct error : std::runtime_error
	{
		error (const std::string &what_arg) : std::runtime_error(what_arg) {}
	};

	// to be thrown when there is some internal error (bug) in the library
	struct internalError : error
	{
		internalError (const std::string &what_arg) : error(what_arg) {}
	};

}


#endif
