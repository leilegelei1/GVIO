//
// Created by jerry on 2021/9/20.
//

#include <baseStruct.h>

void loadKittiData(KittiCalibration& kitti_calibration,
                   vector<ImuMeasurement>& imu_measurements,
                   vector<GpsMeasurement>& gps_measurements,
                   vector<ImageMeasurement>& img_measurements,
                   string imu_metadata_file,
                   string imu_data_file,
                   string gps_data_file,
                   string img_data_file){
    string line;

    // Read IMU metadata and compute relative sensor pose transforms
    // BodyPtx BodyPty BodyPtz BodyPrx BodyPry BodyPrz AccelerometerSigma GyroscopeSigma IntegrationSigma
    // AccelerometerBiasSigma GyroscopeBiasSigma AverageDeltaT

    ifstream imu_metadata(imu_metadata_file.c_str());

    printf("-- Reading sensor metadata\n");

    getline(imu_metadata, line, '\n');  // ignore the first line

    // Load Kitti calibration
    getline(imu_metadata, line, '\n');
    sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
           &kitti_calibration.body_ptx,
           &kitti_calibration.body_pty,
           &kitti_calibration.body_ptz,
           &kitti_calibration.body_prx,
           &kitti_calibration.body_pry,
           &kitti_calibration.body_prz,
           &kitti_calibration.accelerometer_sigma,
           &kitti_calibration.gyroscope_sigma,
           &kitti_calibration.integration_sigma,
           &kitti_calibration.accelerometer_bias_sigma,
           &kitti_calibration.gyroscope_bias_sigma,
           &kitti_calibration.average_delta_t);
    printf("IMU metadata: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
           kitti_calibration.body_ptx,
           kitti_calibration.body_pty,
           kitti_calibration.body_ptz,
           kitti_calibration.body_prx,
           kitti_calibration.body_pry,
           kitti_calibration.body_prz,
           kitti_calibration.accelerometer_sigma,
           kitti_calibration.gyroscope_sigma,
           kitti_calibration.integration_sigma,
           kitti_calibration.accelerometer_bias_sigma,
           kitti_calibration.gyroscope_bias_sigma,
           kitti_calibration.average_delta_t);
    // Read IMU data
    // Time dt accelX accelY accelZ omegaX omegaY omegaZ
    printf("-- Reading IMU measurements from file\n");
    {
        ifstream imu_data(imu_data_file.c_str());
        getline(imu_data, line, '\n');  // ignore the first line

        double time = 0, dt = 0, acc_x = 0, acc_y = 0, acc_z = 0, gyro_x = 0, gyro_y = 0, gyro_z = 0;
        while (!imu_data.eof()) {
            getline(imu_data, line, '\n');
            sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf",
                   &time, &dt,
                   &acc_x, &acc_y, &acc_z,
                   &gyro_x, &gyro_y, &gyro_z);

            ImuMeasurement measurement;
            measurement.time = time;
            measurement.dt = dt;
            measurement.accelerometer = Vector3(acc_x, acc_y, acc_z);
            measurement.gyroscope = Vector3(gyro_x, gyro_y, gyro_z);
            imu_measurements.push_back(measurement);
        }
    }

    // Read GPS data
    // Time,X,Y,Z
    printf("-- Reading GPS measurements from file\n");
    {
        ifstream gps_data(gps_data_file.c_str());
        getline(gps_data, line, '\n');  // ignore the first line

        double time = 0, gps_x = 0, gps_y = 0, gps_z = 0;
        while (!gps_data.eof()) {
            getline(gps_data, line, '\n');
            sscanf(line.c_str(), "%lf %lf %lf %lf", &time, &gps_x, &gps_y, &gps_z);

            GpsMeasurement measurement;
            measurement.time = time;
            measurement.position = Vector3(gps_x, gps_y, gps_z);
            gps_measurements.push_back(measurement);
        }
    }

    printf("-- Reading IMG measurements from file\n");
    {
        ifstream img_data(img_data_file.c_str());
        getline(img_data, line, '\n');  // ignore the first line

        double time = 0;
        char img_p[512] = {0};
        while (!img_data.eof()) {
            getline(img_data, line, '\n');
            sscanf(line.c_str(), "%lf %s", &time, img_p);

            ImageMeasurement measurement;
            measurement.time = time;
            measurement.path = img_p;
            img_measurements.push_back(measurement);
        }
    }
}