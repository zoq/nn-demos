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
  const static std::string url = "http://virtual-artz.de/mnist/index.html";

  // The train dataset to be loaded.
  const static std::string trainDataset = "mnist_train.csv";

  // The test dataset to be loaded.
  const static std::string testDataset = "mnist_test.csv";
}