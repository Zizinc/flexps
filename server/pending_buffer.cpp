#include "server/pending_buffer.hpp"

namespace flexps {

std::vector<Message> PendingBuffer::Pop(int clock) {
  std::vector<Message> poped_msg;
  if (buffer_.find(clock) != buffer_.end()) {
    poped_msg = std::move(buffer_[clock]);
    buffer_.erase(clock);
  }
  return poped_msg;
}

void PendingBuffer::Push(int clock, Message& msg) {
  // Todo(Ruoyu Wu): check the clock passed in is less than the minimum clock in the buffer
  if (buffer_.find(clock) == buffer_.end()) {
    buffer_[clock] = std::vector<Message>({std::move(msg)});
  } else {
    buffer_[clock].push_back(std::move(msg));
  }
}

int PendingBuffer::Size(int progress) { return buffer_[progress].size(); }

}  // namespace flexps