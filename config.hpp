/**
 * @file config.hpp
 * @author Marcus Edel
 *
 * Miscellaneous settings.
 */

#include <iostream>
#include <string>

namespace config {
  // The deployment path.
  const static std::string path = "mnist";

  // The websocket port.
  const static int port = 9972;

  // The deployment url.
  const static std::string url = "ws://urgs.org:9972/mnist";

  // The dataset to be loaded.
  const static std::string dataset = "/home/marcus/src/mlpack/build/mnist_first250_training_4s_and_9s.arm";
}