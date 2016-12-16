/**
 * @file job.hpp
 * @author Marcus Edel
 *
 * Miscellaneous job settings.
 */

#include <iostream>
#include <mutex>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/asio.hpp>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/base64/base64.hpp>

#include <mlpack/core.hpp>

#include <mlpack/core/optimizers/rmsprop/rmsprop.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>

#include "../parser.hpp"
#include "../graphics.hpp"
#include "config.hpp"

namespace job {

  std::mutex jobMutex;

  using namespace mlpack;
  using namespace mlpack::ann;
  using namespace mlpack::optimization;

  // Additional data for every connection.
  struct connection_data {
    int sessionid;
  };

  // Custom config to manage session ids.
  struct custom_config : public websocketpp::config::asio {
    // Pull default settings from our core config.
    typedef websocketpp::config::asio core;

    typedef core::concurrency_type concurrency_type;
    typedef core::request_type request_type;
    typedef core::response_type response_type;
    typedef core::message_type message_type;
    typedef core::con_msg_manager_type con_msg_manager_type;
    typedef core::endpoint_msg_manager_type endpoint_msg_manager_type;
    typedef core::alog_type alog_type;
    typedef core::elog_type elog_type;
    typedef core::rng_type rng_type;
    typedef core::transport_type transport_type;
    typedef core::endpoint_base endpoint_base;

    // Set a custom connection_base class.
    typedef connection_data connection_base;
  };

  typedef websocketpp::server<custom_config> server;
  typedef server::connection_ptr connection_ptr;

  using websocketpp::connection_hdl;
  using websocketpp::lib::placeholders::_1;
  using websocketpp::lib::placeholders::_2;
  using websocketpp::lib::bind;

  /*
   * Class to handle the training process of the specified neural network.
   */
  class JobContainer {
    public:
      /*
       * Create the job object to train the neural network on the specified
       * dataset.
       */
      JobContainer(server& s,
                   connection_hdl hdl,
                   int sessionid,
                   arma::mat& XTrain,
                   arma::mat& YTrain,
                   arma::mat& XTest,
                   arma::mat& YTest) :
        XTrain(XTrain),
        YTrain(YTrain),
        XTest(XTest),
        YTest(YTest),
        s(&s),
        sessionid(sessionid),
        reset(false),
        step(false),
        sampleIndex(0)
      {
        nPointsTrain = XTrain.n_cols;
        nPointsTest = XTest.n_cols;

        // Add connection handle to the list of all connections. We use the list
        // to send all connected clients the data generated through the training
        // process.
        connectionHandles.push_back(hdl);

        convLayer0Ptr = new Convolution<>(1, 8, 5, 5, 1, 1, 0, 0, 28, 28);
        baseLayer0Ptr = new ReLULayer<>();
        poolingLayer0Ptr = new MaxPooling<>(8, 8, 2, 2);
        convLayer1Ptr = new Convolution<>(8, 12, 2, 2);
        poolingLayer1Ptr = new MaxPooling<>(2, 2, 2, 2);
        baseLayer1Ptr = new ReLULayer<>();
        linearLayer0Ptr = new Linear<>(192, 10);
        softmaxLayer0Ptr = new LogSoftMax<>();

        net = new FFN<NegativeLogLikelihood<> >(XTrain, YTrain);
        net->Add(convLayer0Ptr);
        net->Add(baseLayer0Ptr);
        net->Add(poolingLayer0Ptr);
        net->Add(convLayer1Ptr);
        net->Add(baseLayer1Ptr);
        net->Add(poolingLayer1Ptr);
        net->Add(linearLayer0Ptr);
        net->Add(softmaxLayer0Ptr);

        trainingError = 0;
        trainingErrorcurrent = 0;
        trainingSeqCur = 0;
        trainingSeqStart = 0;
        state = 1; // Waiting

        // Initilize the confusion matrix.
        confusion = arma::Mat<int>(10, 10);

        // Initilize the confusion matrix samples.
        confusionClassification = arma::zeros<arma::Mat<int> >(10 * 10, 25);

        // Initilize the confusion matrix probabilities.
        confusionProps = arma::zeros<arma::mat>(10 * 10, 25);

        // Send current job state.
        SendState();
      }

      /*
       * Run the job.
       */
      void Run()
      {
        jobState = 0;
        state = 0; // Running

        // Reset the current paremter.
        if (meanSquaredGradient.empty() || reset)
        {
          sampleIndex = 0;

          if (reset)
          {
            reset = false;
          }
        }

        SendState();

        // Create train and test iteration index.
        arma::Col<size_t> indexTrain = arma::linspace<arma::Col<size_t> >(0,
            nPointsTrain - 1, nPointsTrain);

        arma::Col<size_t> indexTest = arma::linspace<arma::Col<size_t> >(0,
            nPointsTest - 1, nPointsTest);

        // We set the default prediction mode.
        bool predict = true;

        // RMSProp settings.
        const double stepSize = 0.01;
        const double alpha = 0.99;
        const double eps = 1e-8;

        arma::cube moduleOutput;
        arma::cube moduleWeight;
        arma::cube moduleGradient;
        arma::mat funcGradient;

        for (size_t z = 0; z < 5; z++)
        {
          bool predictOnly = true;
          SendState();

          if (state == 2)
          {
            break;
          }

          if (!step)
          {
            confusion.zeros();
            confusionClassification.zeros();
            confusionProps.zeros();
          }

          trainingError = 0;
          testingError = 0;
          predict = true;

          for (size_t i = 0; i < nPointsTrain; i++)
          {
            sampleIndex++;
            SendTrainInfo(z, sampleIndex);

            if (state == 2)
            {
              break;
            }

            if (((i % 500) == 0) || step)
            {
              predict = true;
            }

            indexTrain = arma::shuffle(indexTrain);

            arma::mat currentTarget = YTrain.col(indexTrain(i));

            arma::mat currentInput = XTrain.col(indexTrain(i));
            currentInput.reshape(28, 28);

            SendActivation(currentInput, 0);

            net->Gradient(net->Parameters(), indexTrain(i), funcGradient);

            if ((i % 500) == 0 || step)
            {
              predict = true;
            }

            if ((i % 250) == 0)
            {
              // Send Activations.
              moduleOutput = arma::cube(convLayer0Ptr->OutputParameter().memptr(),
                  std::sqrt(convLayer0Ptr->OutputParameter().n_elem / 8),
                  std::sqrt(convLayer0Ptr->OutputParameter().n_elem / 8),
                  8, false, false);
              moduleOutput = moduleOutput.slices(0, 5);
              SendActivation(moduleOutput, 1);

              moduleOutput = arma::cube(poolingLayer0Ptr->OutputParameter().memptr(),
                  std::sqrt(poolingLayer0Ptr->OutputParameter().n_elem / 8),
                  std::sqrt(poolingLayer0Ptr->OutputParameter().n_elem / 8),
                  8, false, false);
              moduleOutput = moduleOutput.slices(0, 5);
              SendActivation(moduleOutput, 7);

              moduleOutput = arma::cube(convLayer1Ptr->OutputParameter().memptr(),
                  std::sqrt(convLayer1Ptr->OutputParameter().n_elem / 12),
                  std::sqrt(convLayer1Ptr->OutputParameter().n_elem / 12),
                  12, false, false);
              moduleOutput = moduleOutput.slices(0, 9);
              SendActivation(moduleOutput, 13);

              moduleOutput = arma::cube(poolingLayer1Ptr->OutputParameter().memptr(),
                  std::sqrt(poolingLayer1Ptr->OutputParameter().n_elem / 12),
                  std::sqrt(poolingLayer1Ptr->OutputParameter().n_elem / 12),
                  12, false, false);
              moduleOutput = moduleOutput.slices(0, 9);
              SendActivation(moduleOutput, 23);

              arma::colvec outputTemp = softmaxLayer0Ptr->OutputParameter().col(0);
              outputTemp = 1 - arma::normalise(arma::abs(outputTemp));
              SendActivation(outputTemp, 33);

              // Send Weights.
              moduleWeight = arma::cube(convLayer0Ptr->Parameters().memptr(),
                  5, 5, 8, false, false);
              moduleWeight = moduleWeight.slices(0, 5);
              SendWeight(moduleWeight, 1);

              moduleWeight = arma::cube(convLayer1Ptr->Parameters().memptr(),
                  2, 2, 8 * 12, false, false);
              SendWeight(moduleWeight, 7);

              // Send Gradients.
              moduleGradient = arma::cube(convLayer0Ptr->Gradient().memptr(),
                  5, 5, 8, false, false);
              moduleGradient = moduleGradient.slices(0, 5);
              SendGradient(moduleGradient, 1);

              moduleGradient = arma::cube(convLayer1Ptr->Gradient().memptr(),
                  2, 2, 8 * 12, false, false);
              SendGradient(moduleGradient, 7);
            }

            if (meanSquaredGradient.empty())
            {
              meanSquaredGradient = funcGradient;
              meanSquaredGradient.zeros();
            }

            arma::mat trainPredictionOutput =
                  softmaxLayer0Ptr->OutputParameter();

            if (trainPredictionOutput.is_finite())
            {
              arma::uword trainPredictionIndexMax, trainTargetIndexMax;

              trainPredictionOutput.max(trainPredictionIndexMax);
              trainTargetIndexMax = int(YTrain(indexTrain(i))) - 1;

              confusion(trainTargetIndexMax, trainPredictionIndexMax)++;

              // Get the sample class.
              arma::uword confusionClassIndex = sub2ind(10, 10,
                  trainTargetIndexMax, trainPredictionIndexMax);

              // Get the current sample index.
              arma::uword confusionSampleIndex = confusionClassification(
                  confusionClassIndex, 0);

              // Set the current sample.
              confusionClassification(confusionClassIndex,
                (confusionSampleIndex % 18) + 1) = indexTrain(i);

              // Set the current prediction propability.
              confusionProps(confusionClassIndex,
                  (confusionSampleIndex % 18) + 1) = trainPredictionOutput(
                  trainPredictionIndexMax);

              // Increase the sample index.
              confusionClassification(confusionClassIndex, 0)++;
              confusionProps(confusionClassIndex, 0)++;
            }

            if (predict || ((i % 250) == 0))
            {
              SendConfusion(confusion);
            }

            if (predict)
            {
              predict = false;
              indexTest = arma::shuffle(indexTest);

              for (size_t k = 0; k < 300; k++)
              {
                arma::mat predictionTemp;
                arma::mat predictionInput = XTest.col(indexTest(k));
                net->Predict(predictionInput, predictionTemp);

                if (k < 11)
                {
                  arma::mat predictionSample = arma::mat(
                      XTest.colptr(indexTest(k)), 28, 28, false, false);

                  arma::uvec idx = arma::sort_index(predictionTemp.col(0),
                      "descend");

                  predictionInfo[k].image = predictionSample;

                  arma::mat probNorm = 1 - arma::normalise(arma::abs(
                      predictionTemp));

                  for (size_t j = 0; j < 5; j++)
                  {
                    predictionInfo[k].prediction[j] = std::to_string(idx(j));
                    predictionInfo[k].prob[j] = std::to_string(
                        probNorm(idx(j), 0));
                  }

                  predictionInfo[k].target = std::to_string(
                      int(YTest(indexTest(k))) - 1);
                }
              }

              // Send collected prediction data to all connected clients.
              SendPrediction();

              if (z != 0)
              {
                predictOnly = false;
              }
            }

            meanSquaredGradient *= alpha;
            meanSquaredGradient += (1 - alpha) * (funcGradient % funcGradient);
                net->Parameters() -= stepSize * funcGradient /
                    (arma::sqrt(meanSquaredGradient) + eps);

            if (step)
            {
              state = 2;
              step = false;
              break;
            }
          }

          SendTrainingStatus(z);
        }

        state = 2; // Stop
        SendState();
        jobState = 1;
      }

      //! Get the state.
      int State() const { return state; }
      //! Modify the state.
      int& State() { return state; }

      //! Get the reset mode.
      bool Reset() const { return reset; }
      //! Modify the reset mode.
      bool& Reset() { return reset; }

      //! Get the step mode.
      bool Step() const { return step; }
      //! Modify the step mode.
      bool& Step() { return step; }

      /*
       * Send the confusion samples of the given id to all connected clients.
       */
      void ConfusionSamples(int id)
      {
        m.lock();

        if (confusionClassification.empty())
        {
          m.unlock();
          return;
        }

        size_t classIndex = 0;

        if (id <= 9)
        {
          classIndex = sub2ind(10, 10, id, 0);
        }
        else
        {
          std::string idStr = std::to_string(id);
          std::string rowStr(1, idStr.at(1));
          std::string colStr(1, idStr.at(0));

          classIndex = sub2ind(10, 10, std::stoi(rowStr), std::stoi(colStr));
        }

        for (size_t j = 1, sampleId = 0; j < confusionClassification.n_cols; j++)
        {
          if (confusionClassification(classIndex, j) != 0)
          {
            arma::mat input = arma::mat(XTrain.colptr(
                confusionClassification(classIndex, j)), 28, 28, false, false);

            std::stringstream samplePropConfusion;
            samplePropConfusion << std::fixed << std::setprecision(2);
            samplePropConfusion << confusionProps(classIndex, j);

            std::string output = "{";
            output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
            output += "\"samplePropConfusion" + std::to_string(sampleId) + "\": \"" + samplePropConfusion.str() + "\",";
            std::string imageString = graphics::Mat2Image(arma::normalise(input) * 255);
            output += "\"sampleConfusion" + std::to_string(sampleId++) + "\": \"";
            output += websocketpp::base64_encode(imageString) + "\"";
            output += "}";

            size_t i = 0;
            for (; i < connectionHandles.size(); i++)
            {
              try {
                s->send(connectionHandles[i], output,
                    websocketpp::frame::opcode::TEXT);
              } catch (...) {
                // Remove connection handle from the connection list.
                connectionHandles.erase(connectionHandles.begin() + i);
              }
            }
          }
        }

        m.unlock();
      }

      /*
       * Get the continues index of the given row col.
       */
      size_t sub2ind(const size_t rows,
                     const size_t cols,
                     const size_t row,
                     const size_t col)
      {
         return row*cols+col;
      }

      /*
       * Add a connection handle to the list of all connections managed by this
       * job. We use the connection list to send the training state to all
       * connected clients.
       */
      void AddConnectionHandle(connection_hdl hdl)
      {
        connectionHandles.push_back(hdl);
        SendState();
      }

      /*
       * Send the current state to all connected clients.
       */
      void SendState()
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"state\": \"" + std::to_string(state) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      int jobState;

    private:
      /*
       * Send the current training and testing info to all connected clients.
       */
      void SendTrainInfo(size_t iteration, size_t sampleIndex)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream trainingErrorString;
        trainingErrorString << std::fixed << std::setprecision(2);
        trainingErrorString << trainingErrorcurrent;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"sampleIndex\": \"" + std::to_string(sampleIndex) + "\",";
        output += "\"classificationLoss\": \"" + trainingErrorString.str() + "\",";
        output += "\"iteration\": \"" + std::to_string(iteration) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current training state to all connected clients.
       */
      void SendTrainingStatus(size_t z)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream trainingErrorString;
        trainingErrorString << std::fixed << std::setprecision(2);
        trainingErrorString << trainingErrorcurrent;

        std::stringstream testingErrorString;
        testingErrorString << std::fixed << std::setprecision(2);
        testingErrorString << testingErrorcurrent;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"trainingError\": \"" + trainingErrorString.str() + "\",";
        output += "\"testingError\": \"" + testingErrorString.str() + "\",";
        output += "\"iteration\": \"" + std::to_string(z) + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current confusion matrix to all connected clients.
       */
      void SendConfusion(arma::Mat<int>& confusion)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::stringstream consufionString;
        for (size_t i = 0; i < confusion.n_elem; i++)
        {
          consufionString << confusion(i);

          if (i != (confusion.n_elem - 1))
            consufionString << ";";
        }

        std::stringstream precisionString;
        for (size_t i = 0; i < confusion.n_rows; i++)
        {
          double tp = confusion(i, i);
          double fp = arma::accu(confusion.row(i)) - tp;

          double precision = tp / (tp + fp);

          if (precision != precision)
            precision = 0;

          precisionString << std::fixed << std::setprecision(2);
          precisionString << precision << ";";
        }

        std::stringstream recallString;
        for (size_t i = 0; i < confusion.n_rows; i++)
        {
          double tp = confusion(i, i);
          double fn = arma::accu(confusion.col(i)) - tp;

          double recall = tp / (tp + fn);

          if (recall != recall)
            recall = 0;

          recallString << std::fixed << std::setprecision(2);
          recallString << recall << ";";
        }

        double accuracy = (double) arma::accu(confusion.diag()) /
            (double) arma::accu(confusion);

        if (accuracy != accuracy)
          accuracy = 0;

        std::stringstream accuracyString;
        accuracyString << std::fixed << std::setprecision(2);
        accuracyString << accuracy;

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
        output += "\"precision\": \"" + precisionString.str() + "\",";
        output += "\"recall\": \"" + recallString.str() + "\",";
        output += "\"accuracy\": \"" + accuracyString.str() + "\",";
        output += "\"confusion\": \"" + consufionString.str() + "\"";
        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current prediction to all connected clients.
       */
      void SendPrediction()
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < 11; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(predictionInfo[i].image) * 255);
          output += "\"predictionActivation" + std::to_string(i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";

          output += "\"predictionValues" + std::to_string(i) + "\": \"";
          std::stringstream predictionValues;
          predictionValues << std::fixed << std::setprecision(2);
          for (size_t j = 0; j < 5; j++)
          {
            predictionValues << predictionInfo[i].prediction[j] << "-";
          }
          predictionValues << predictionInfo[i].target;
          output += predictionValues.str() + "\",";

          output += "\"probValues" + std::to_string(i) + "\": \"";
          std::stringstream probValues;
          probValues << std::fixed << std::setprecision(2);
          for (size_t j = 0; j < 5; j++)
          {
            probValues << predictionInfo[i].prob[j] << "-";
          }
          probValues << predictionInfo[i].target;
          output += probValues.str() + "\",";
        }

        output += "\"state\": \"" + std::to_string(state) + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::mat& m, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        std::string imageString = graphics::Mat2Image(arma::normalise(m) * 255);
        output += "\"layerActivation" + std::to_string(id) + "\": \"";
        output += websocketpp::base64_encode(imageString) + "\",";

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << m.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << m.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::colvec& v, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < v.n_elem; i++)
        {
          arma::mat vTemp(1, 1);
          vTemp.fill(v(i));
          std::string imageString = graphics::Mat2Image(vTemp * 255);
          output += "\"layerActivation" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << v.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << v.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current activation to all connected clients.
       */
      void SendActivation(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);

          output += "\"layerActivation" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxActivation" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minActivation" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current weight to all connected clients.
       */
      void SendWeight(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);

          output += "\"layerWeight" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxWeight" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minWeight" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      /*
       * Send the current gradient to all connected clients.
       */
      void SendGradient(arma::cube& c, int id)
      {
        if (connectionHandles.size() == 0)
        {
          return;
        }

        std::string output = "{";
        output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";

        for (size_t i = 0; i < c.n_slices; i++)
        {
          std::string imageString = graphics::Mat2Image(
              arma::normalise(c.slice(i)) * 255);

          output += "\"layerGradient" + std::to_string(id + i) + "\": \"";
          output += websocketpp::base64_encode(imageString) + "\",";
        }

        std::stringstream maxActivation;
        maxActivation << std::fixed << std::setprecision(2) << c.max();
        output += "\"maxGradient" + std::to_string(id) + "\": \"";
        output += maxActivation.str() + "\",";

        std::stringstream minActivation;
        minActivation << std::fixed << std::setprecision(2) << c.min();
        output += "\"minGradient" + std::to_string(id) + "\": \"";
        output += minActivation.str() + "\"";

        output += "}";

        size_t i = 0;
        for (; i < connectionHandles.size(); i++)
        {
          try {
            s->send(connectionHandles[i], output,
                websocketpp::frame::opcode::TEXT);
          } catch (...) {
            // Remove connection handle from the connection list.
            connectionHandles.erase(connectionHandles.begin() + i);
          }
        }
      }

      // Structure used to manage the predictions.
      struct PredictionInfo
      {
        arma::mat image;
        std::string prob[5];
        std::string prediction[5];
        std::string target;
      };

      // The job state (running = 0, queued = 1, stopped = 2).
      int state;

      // The current error and the overall training error.
      double trainingError, trainingErrorcurrent;

      // The current error and the overall testing error.
      double testingError, testingErrorcurrent;

      // The current training sequence and the overall training sequence.
      size_t trainingSeqStart, trainingSeqCur;

      // The prediction info (matrix, probability and prediction.
      PredictionInfo predictionInfo[11];

      // The jon content.
      std::string content;

      // The train and test set.
      arma::mat YTrain, YTest, XTrain, XTest;

      // The number of points in the dataset.
      arma::uword nPointsTrain, nPointsTest;

      // The gradient of the first  and second convolution layer.
      arma::cube gradient0, gradient1;

      // The network structure.
      Convolution<>* convLayer0Ptr;
      ReLULayer<>* baseLayer0Ptr;
      MaxPooling<>* poolingLayer0Ptr;
      Convolution<>* convLayer1Ptr;
      ReLULayer<>* baseLayer1Ptr;
      MaxPooling<>* poolingLayer1Ptr;
      Linear<>* linearLayer0Ptr;
      LogSoftMax<>* softmaxLayer0Ptr;

      //! The current network parameter;
      arma::mat networkParameter;

      arma::mat meanSquaredGradient;

      FFN<NegativeLogLikelihood<> >* net;

      //! Locally stored server object used for sending.
      server* s;

      //! List of connection handles.
      std::vector<connection_hdl> connectionHandles;

      //! Locally stored session id.
      int sessionid;

      //! Locally-stored confusion matrix.
      arma::Mat<int> confusion;

      //! Locally-stored confusion classification matrix.
      arma::Mat<int> confusionClassification;

      //! Locally-stored confusion probabilities matrix.
      arma::mat confusionProps;

      std::mutex m;

      //! Locally-stored reset training process.
      bool reset;

      //! Locally-stored step parameter.
      bool step;

      //! Locally-stored sample index parameter.
      size_t sampleIndex;
};

  /*
   * Redirect the user to the initial page.
   *
   * @param id The user/job id.
   */
  std::string InitNewPage(const int id)
  {
    std::string output = "{";
    output += "\"$schema\": \"http://json-schema.org/draft-04/schema#\",";
    output += "\"redirect\": \"" + config::url + "?id=";
    output += std::to_string(id) + "\"";
    output += "}";

    return output;
  }

  /*
   * Job wrapper class that holds the actual job.
   */
  class JobWrapper {
    public:
      /*
       * Create the JobWrapper object and store the specified job.
       */
      JobWrapper(JobContainer& jobContainer) : jobContainer(&jobContainer)
      {
        /* Nothing to do here */
      }

      /*
       * Dispatch the job.
       */
      void operator()(){
        jobContainer->Run();
        return;
      }

    public:
      //! Locally stored pointer of an job object.
      JobContainer* jobContainer;
  };

  boost::ptr_vector<JobContainer> jobQueue;
  boost::asio::io_service io_service;

  class Server {
    public:
      Server() : m_next_sessionid(1) {
        m_server.init_asio();

        m_server.set_open_handler(bind(&Server::on_open,this,_1));
        m_server.set_close_handler(bind(&Server::on_close,this,_1));
        m_server.set_message_handler(bind(&Server::on_message,this,_1,_2));
        m_server.set_access_channels(websocketpp::log::alevel::none);
        m_server.set_error_channels(websocketpp::log::alevel::fail);

        arma::mat X;
        // Load the train dataset.
        X.load(config::trainDataset);

        if (X.n_cols >= 784 && X.n_cols <= 785)
        {
          X = X.t();
        }

        nPointsTrain = X.n_cols;

        // Build the target matrix for the train dataset.
        YTrain = arma::zeros<arma::mat>(1, nPointsTrain);
        for (size_t i = 0; i < nPointsTrain; i++)
        {
          size_t target = X(0, i);
          YTrain(i) = target + 1;
        }

        X.shed_row(0);

        XTrain = arma::mat(28 * 28, nPointsTrain);
        // Normalize each point since these are images of the train dataset.
        for (arma::uword i = 0; i < nPointsTrain; i++)
        {
          XTrain.col(i) = X.col(i) / norm(X.col(i), 2);
        }

        // Load the test dataset.
        X.load(config::testDataset);

        if (X.n_cols >= 784 && X.n_cols <= 785)
        {
          X = X.t();
        }

        nPointsTest = X.n_cols;

        // Build the target matrix for the test dataset.
        YTest = arma::zeros<arma::mat>(1, nPointsTest);
        for (size_t i = 0; i < nPointsTest; i++)
        {
          size_t target = X(0, i);
          YTest.col(i) = target + 1;
        }

        X.shed_row(0);

        XTest = arma::mat(28 * 28, nPointsTest);
        // Normalize each point since these are images of the test dataset.
        for (arma::uword i = 0; i < nPointsTest; i++)
        {
          XTest.col(i) = X.col(i) / norm(X.col(i), 2);
        }
      }

      void on_open(connection_hdl hdl) {
        connection_ptr con = m_server.get_con_from_hdl(hdl);

        con->sessionid = m_next_sessionid++;
      }

      void on_close(connection_hdl hdl) {

          connection_ptr con = m_server.get_con_from_hdl(hdl);
      }

      void on_message(connection_hdl hdl, server::message_ptr msg) {
        connection_ptr connection = m_server.get_con_from_hdl(hdl);


        const std::string& host = connection->get_host();
        const std::string& resource = connection->get_resource();

        std::cout << "host: " << host << " resource: " << resource << std::endl;

        // Parse the request resource and extract the event and the event id.
        parser::info uriInfo;
        parser::parseURI(resource, uriInfo);

        parser::info eventInfo;
        parser::parseURI(msg->get_payload(), eventInfo);

        std::cout << "id: " << uriInfo.id << std::endl;
        std::cout << "event: " << eventInfo.event << std::endl;
        std::cout << "event-id: " << eventInfo.eventId << std::endl;

        // Handle the event.
        if (uriInfo.id < 0 || (uriInfo.id + 1) > jobQueue.size() )
        {
          // Create a new task.
          jobQueue.push_back(new JobContainer(m_server, hdl,
              connection->sessionid, XTrain, YTrain, XTest, YTest));
          io_service.dispatch(boost::move(
              JobWrapper(job::jobQueue.back())));

          // Redirect the user to the initial page using the new job id.
          m_server.send(hdl, InitNewPage(jobQueue.size() - 1),
              websocketpp::frame::opcode::TEXT);
        }
        else if(eventInfo.event == "confusion" && eventInfo.eventId >= 0)
        {
          jobQueue[uriInfo.id].ConfusionSamples(eventInfo.eventId);
        }
        else if(eventInfo.event == "step" && eventInfo.eventId >= 0)
        {
          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }

          state = jobQueue[uriInfo.id].State();

          jobQueue[uriInfo.id].Step() = true;

          if (state == 2)
          {
            // Queue the job to be running next.
            jobQueue[uriInfo.id].State() = 1;

            if (jobQueue[uriInfo.id].jobState == 1)
            {
              io_service.dispatch(boost::move(JobWrapper(jobQueue[uriInfo.id])));
            }
            else
            {
              jobQueue[uriInfo.id].State() = 0;
            }
          }

          jobQueue[uriInfo.id].SendState();
        }
        else if(eventInfo.event == "reset" && eventInfo.eventId >= 0)
        {
          jobQueue[uriInfo.id].Reset() = true;

          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }

          state = jobQueue[uriInfo.id].State();
          jobQueue[uriInfo.id].Reset() = true;

          if (state == 2)
          {
            // Queue the job to be running next.
            jobQueue[uriInfo.id].State() = 1;

            if (jobQueue[uriInfo.id].jobState == 1)
            {
              io_service.dispatch(boost::move(JobWrapper(jobQueue[uriInfo.id])));
            }
            else
            {
              jobQueue[uriInfo.id].State() = 0;
            }
          }

          jobQueue[uriInfo.id].SendState();
        }
        else if(eventInfo.event == "start" && eventInfo.eventId >= 0)
        {
          // Set the correct job state.
          int state = jobQueue[uriInfo.id].State();
          if (state == 0)
          {
             // Stop the job, because it is already running.
            jobQueue[uriInfo.id].State() = 2;
          }
          else if (state == 2)
          {
            // Queue the job to be running next.
            jobQueue[uriInfo.id].State() = 1;

            if (jobQueue[uriInfo.id].jobState == 1)
            {
              io_service.dispatch(boost::move(JobWrapper(jobQueue[uriInfo.id])));
            }
            else
            {
              jobQueue[uriInfo.id].State() = 0;
            }
          }

          jobQueue[uriInfo.id].SendState();
        }
        else
        {
          jobQueue[uriInfo.id].AddConnectionHandle(hdl);
          jobQueue[uriInfo.id].SendState();
        }
      }

      void run(uint16_t port) {
        m_server.listen(port);
        m_server.start_accept();
        m_server.run();
      }

    private:
      int m_next_sessionid;
      server m_server;

      arma::mat YTrain, YTest, XTrain, XTest;
      arma::uword nPointsTrain, nPointsTest;
  };
}