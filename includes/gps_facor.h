//
// Created by jerry on 2021/9/19.
//

#ifndef GVIO_GPS_FACOR_H
#define GVIO_GPS_FACOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Testable.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include <string>

namespace gtsam {

    /**
     * A class for a soft prior on any POINT3 type
     * @addtogroup SLAM
     */
    template<class POSE>
    class GpsFactor: public NoiseModelFactor1<POSE> {

    public:
        typedef POSE T;

    public:

        typedef NoiseModelFactor1<POSE> Base;

        Point3 prior_; /** The measurement */

        /** concept check by type */
        GTSAM_CONCEPT_TESTABLE_TYPE(T)

    public:

        /// shorthand for a smart pointer to a factor
        typedef typename boost::shared_ptr<GpsFactor<POSE> > shared_ptr;

        /// Typedef to this class
        typedef GpsFactor<POSE> This;

        /** default constructor - only use for serialization */
        GpsFactor() {}

        ~GpsFactor() override {}

        /** Constructor */
        GpsFactor(Key key, const Point3& prior, const SharedNoiseModel& model = nullptr) :
                Base(model, key), prior_(prior) {
        }

        /** Convenience constructor that takes a full covariance argument */
        GpsFactor(Key key, const Point3& prior, const Matrix& covariance) :
                Base(noiseModel::Gaussian::Covariance(covariance), key), prior_(prior) {
        }

        /// @return a deep copy of this factor
        gtsam::NonlinearFactor::shared_ptr clone() const override {
            return boost::static_pointer_cast<gtsam::NonlinearFactor>(
                    gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

        /** implement functions needed for Testable */

        /** print */
        void print(const std::string& s,
                   const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override {
            std::cout << s << "PriorFactor on " << keyFormatter(this->key()) << "\n";
            traits<Point3>::Print(prior_, "  prior mean: ");
            if (this->noiseModel_)
                this->noiseModel_->print("  noise model: ");
            else
                std::cout << "no noise model" << std::endl;
        }

        /** equals */
        bool equals(const NonlinearFactor& expected, double tol=1e-9) const override {
            const This* e = dynamic_cast<const This*> (&expected);

            return e != nullptr && Base::equals(*e, tol) && traits<Point3>::Equals(prior_, e->prior(), tol);
        }

        /** implement functions needed to derive from Factor */

        /** vector of errors */
        Vector evaluateError(const POSE & x, boost::optional<Matrix&> H = boost::none) const override {
            // manifold equivalent of z-x -> Local(x,z)
            // TODO(ASL) Add Jacobians.
            return x.translation(H) - prior_;
        }

        const Point3 & prior() const { return prior_; }

    private:

        /** Serialization function */
        friend class boost::serialization::access;
        template<class ARCHIVE>
        void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
            ar & boost::serialization::make_nvp("NoiseModelFactor1",
                                                boost::serialization::base_object<Base>(*this));
            ar & BOOST_SERIALIZATION_NVP(prior_);
        }

        // Alignment, see https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
        enum { NeedsToAlign = (sizeof(T) % 16) == 0 };
    public:
        GTSAM_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
    };

    /// traits
    template<class POINT3>
    struct traits<GpsFactor<POINT3> > : public Testable<GpsFactor<POINT3> > {};
}

#endif //GVIO_GPS_FACOR_H
