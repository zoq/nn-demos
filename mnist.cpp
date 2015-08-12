/**
 * @file mnist.hpp
 * @author Marcus Edel
 *
 * Demo to visualize the training process of a convolutional neural network.
 */

#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <iostream>

#include "job.hpp"

int main(int argc, char **argv)
{
  boost::thread_group threads;
  std::auto_ptr<boost::asio::io_service::work> work(
      new boost::asio::io_service::work(job::io_service));

  // Spawn worker threads.
  size_t cores = std::thread::hardware_concurrency();
  for (size_t i = 0; i < cores; i++)
  {
    threads.create_thread(boost::bind(
        &boost::asio::io_service::run, &job::io_service));
  }

  // Start the websocket server.
  job::Server server;
  server.run(config::port);

  return 0;
}
