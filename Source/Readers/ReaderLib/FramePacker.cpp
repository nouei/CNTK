//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <algorithm>
#include "FramePacker.h"

namespace Microsoft { namespace MSR { namespace CNTK {


MBLayoutPtr FramePacker::CreateMBLayout(const StreamBatch& batch)
{
    auto violation = find_if(batch.begin(), batch.end(), [](const SequenceDataPtr& s){ return s->m_numberOfSamples > 1; });
    if (violation != batch.end())
    {
        RuntimeError("Detected a non-frame sequence of size %d in frame mode.", 
            (int)(*violation)->m_numberOfSamples);
    }
    // Creating the minibatch layout.
    MBLayoutPtr pMBLayout = make_shared<MBLayout>();
    pMBLayout->InitAsFrameMode(batch.size());
    return pMBLayout;
}

Sequences FramePacker::GetNextSequences()
{
    // In frame mode we know exactly how many samples we want to fetch.
    // So we do not stop till the end of epoch is reached, or we fetch as many samples as needed.
    bool shouldAddOneSample = m_minibatchSize % m_config.m_numberOfWorkers > m_config.m_workerRank;
    int localMinibatchSize = (int)m_minibatchSize / (int)m_config.m_numberOfWorkers + (shouldAddOneSample ? 1 : 0);

    Sequences result;
    result.m_data.resize(m_inputStreamDescriptions.size());
    while (localMinibatchSize > 0 && !result.m_endOfEpoch)
    {
        auto s = m_sequenceEnumerator->GetNextSequences(localMinibatchSize);
        localMinibatchSize -= (int)s.m_data.begin()->size();
        result.m_endOfEpoch = s.m_endOfEpoch;
        for (size_t i = 0; i < s.m_data.size(); ++i)
            result.m_data[i].insert(result.m_data[i].end(), s.m_data[i].begin(), s.m_data[i].end());
    }
    return result;
}

} } }
