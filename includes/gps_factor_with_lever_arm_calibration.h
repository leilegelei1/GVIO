//
// Created by jerry on 2021-09-30.
//

#ifndef GVIO_GPS_FACTOR_WITH_LEVER_ARM_CALIBRATION_H
#define GVIO_GPS_FACTOR_WITH_LEVER_ARM_CALIBRATION_H


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
    template<class POSE,class LEVERARM>
    class GpsFactorWithLeverArmCalibration: public NoiseModelFactor2<POSE,LEVERARM> {

    public:
        typedef NoiseModelFactor2<POSE,LEVERARM> Base;

        Point3 prior_; /** The measurement */

    public:

        /// Typedef to this class
        typedef GpsFactorWithLeverArmCalibration<POSE,LEVERARM> This;

        /** default constructor - only use for serialization */
        GpsFactorWithLeverArmCalibration() {}

        ~GpsFactorWithLeverArmCalibration() override {}

        /** Constructor */
        GpsFactorWithLeverArmCalibration(Key tvecKey,Key leverArmKey, const Point3& prior, const SharedNoiseModel& model = nullptr) :
                Base(model, tvecKey,leverArmKey), prior_(prior) {
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
        Vector evaluateError(const Pose3 & x, const Point3 & l, boost::optional<Matrix&> H1 = boost::none,boost::optional<Matrix&> H1 = boost::none) const override {
            // manifold equivalent of z-x -> Local(x,z)
            // TODO(ASL) Add Jacobians.
            Point3 rtk_pose = x.transformFrom(l,H1,H2);
            gtsam::Matrix H0 = gtsam::Matrix3::Identity();
            *H1 = H0 * (*H1);
            *H2 = H0 * (*H2);
            return rtk_pose - prior_;
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
}

#endif //GVIO_GPS_FACTOR_WITH_LEVER_ARM_CALIBRATION_H
