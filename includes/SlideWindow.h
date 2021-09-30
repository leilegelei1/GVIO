//
// Created by jerry on 2021-09-26.
//

#ifndef GVIO_SLIDEWINDOW_H
#define GVIO_SLIDEWINDOW_H

#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>

namespace gtsam
{
    class SlideWindow : public BatchFixedLagSmoother{
    public:

        bool withMarginal_ = true;

        /** default constructor */
        SlideWindow(double smootherLag = 0.0,bool withMarginal = true, const LevenbergMarquardtParams& parameters = LevenbergMarquardtParams(),
                    bool enforceConsistency = true) :
                BatchFixedLagSmoother(smootherLag,parameters,enforceConsistency),withMarginal_(withMarginal) { };

        /** destructor */
        ~SlideWindow() override { };

        Result update_with_marg_keys(const NonlinearFactorGraph& newFactors = NonlinearFactorGraph(),
                      const Values& newTheta = Values(),
                      const KeyVector& marginalizableKeys = KeyVector(),
//                      const KeyTimestampMap& timestamps = KeyTimestampMap(), TODO 我们不需要时间 这个东西的滑窗我们自己维护
                      const FactorIndices& factorsToRemove = FactorIndices())
        {

            // Update all of the internal variables with the new information
            gttic(augment_system);
            // Add the new variables to theta
            theta_.insert(newTheta);
            // Add new variables to the end of the ordering
            for (const auto key_value : newTheta) {
                ordering_.push_back(key_value.key);
            }
            // Augment Delta
            delta_.insert(newTheta.zeroVectors());

            // Add the new factors to the graph, updating the variable index
            insertFactors(newFactors);
            gttoc(augment_system);

            // remove factors in factorToRemove
            for(const size_t i : factorsToRemove){
                if(factors_[i])
                    factors_[i].reset();
            }

            // Reorder
            gttic(reorder);
            reorder(marginalizableKeys);
            gttoc(reorder);

            // Optimize
            gttic(optimize);
            Result result;
            if (factors_.size() > 0) {
                result = optimize();
            }
            gttoc(optimize);

            // Marginalize out old variables.
            gttic(marginalize);
            if (marginalizableKeys.size() > 0) {
                marginalize_inside(marginalizableKeys);
            }
            gttoc(marginalize);

            return result;
        }


        /** Marginalize out selected variables */
        void marginalize_inside(const KeyVector& marginalizeKeys)
        {
            // In order to marginalize out the selected variables, the factors involved in those variables
            // must be identified and removed. Also, the effect of those removed factors on the
            // remaining variables needs to be accounted for. This will be done with linear container factors
            // from the result of a partial elimination. This function removes the marginalized factors and
            // adds the linearized factors back in.

            // Identify all of the factors involving any marginalized variable. These must be removed.
            set<size_t> removedFactorSlots;
            const VariableIndex variableIndex(factors_);
            for(Key key: marginalizeKeys) {
                const auto& slots = variableIndex[key];
                removedFactorSlots.insert(slots.begin(), slots.end());
            }

            // Add the removed factors to a factor graph
            NonlinearFactorGraph removedFactors;
            for(size_t slot: removedFactorSlots) {
                if (factors_.at(slot)) {
                    removedFactors.push_back(factors_.at(slot));
                }
            }

            // Calculate marginal factors on the remaining keys
            NonlinearFactorGraph marginalFactors = CalculateMarginalFactors(
                    removedFactors, theta_, marginalizeKeys, parameters_.getEliminationFunction());

            // Remove marginalized factors from the factor graph
            removeFactors(removedFactorSlots);

            // Remove marginalized keys from the system
            eraseKeysWithoutTime(marginalizeKeys);

            // Insert the new marginal factors
            if(withMarginal_)
                insertFactors(marginalFactors);
        }

        void eraseKeysWithoutTime(const KeyVector& keys) {

            for(Key key: keys) {
                // Erase the key from the values
                theta_.erase(key);

                // Erase the key from the factor index
                factorIndex_.erase(key);

                // Erase the key from the set of linearized keys
                if (linearKeys_.exists(key)) {
                    linearKeys_.erase(key);
                }
            }

            // Remove marginalized keys from the ordering and delta
            for(Key key: keys) {
                ordering_.erase(find(ordering_.begin(), ordering_.end(), key));
                delta_.erase(key);
            }
        }

        /* ************************************************************************* */
        static GaussianFactorGraph CalculateMarginalFactors(
                const GaussianFactorGraph& graph, const KeyVector& keys,
                const GaussianFactorGraph::Eliminate& eliminateFunction) {
            if (keys.size() == 0) {
                // There are no keys to marginalize. Simply return the input factors
                return graph;
            } else {
                // .first is the eliminated Bayes tree, while .second is the remaining factor graph
                for(auto key : keys)
                {

                }
                return *graph.eliminatePartialMultifrontal(keys, eliminateFunction).second;
            }
        }

        /* ************************************************************************* */
        static NonlinearFactorGraph CalculateMarginalFactors(
                const NonlinearFactorGraph& graph, const Values& theta, const KeyVector& keys,
                const GaussianFactorGraph::Eliminate& eliminateFunction) {
            if (keys.size() == 0) {
                // There are no keys to marginalize. Simply return the input factors
                return graph;
            } else {
                // Create the linear factor graph
                const auto linearFactorGraph = graph.linearize(theta);

                const auto marginalLinearFactors =
                        CalculateMarginalFactors(*linearFactorGraph, keys, eliminateFunction);

                // Wrap in nonlinear container factors
                return LinearContainerFactor::ConvertLinearGraph(marginalLinearFactors, theta);
            }
        }
    };
}

#endif //GVIO_SLIDEWINDOW_H
