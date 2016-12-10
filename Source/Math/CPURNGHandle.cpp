//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "stdafx.h"
#include "CPURNGHandle.h"

namespace Microsoft { namespace MSR { namespace CNTK {

CPURNGHandle::CPURNGHandle(int deviceId, uint64_t seed, uint64_t offset)
    : RNGHandle(deviceId)
{
    m_generator.reset(new std::mt19937_64());
#if defined(_MSC_VER) && (_MSC_VER <= 1800 /*VS2013*/)
    // Up until VS 2015, mt19937_64 takes a 32 bit seed, 
    // see https://connect.microsoft.com/VisualStudio/feedback/details/779231
    m_generator->seed((unsigned long)seed);
#else
    m_generator->seed(seed);
#endif
    m_generator->discard(offset);
}

}}}
