#pragma once

#include <memory>
#include "arrow/buffer.h"
#include "render/engine/common/device.h"


namespace zilliz {
namespace store {

//TODO::tmp objectID
using ObjectID = int32_t;
using Buffer = arrow::Buffer;
using BufferPtr = std::shared_ptr<Buffer>;


struct ObjectBuffer {

    // The data buffer.
    BufferPtr data;

    BufferPtr reserved;

    // The device number.
    int device_num;
};

using ObjectBufferPtr = std::shared_ptr<ObjectBuffer>;


struct ObjectStoreEvent {
    enum class Type {
        kSeal = 0,
        kDelete,
        Type_Num
    };

    Type type;
    ObjectID obj_id;
};

using ObjectStoreEventPtr = std::shared_ptr<ObjectStoreEvent>;


class ObjectStore {
 public:
    virtual ~ObjectStore() {}

    // Connect to the local object store.
    //
    // @param store_socket_name The name of the UNIX domain socket to use to
    //        connect to the local object store.
    // @param num_retries number of attempts to connect to IPC socket, default 50

    virtual void
    Connect(const std::string &store_socket_name,
            int num_retries) = 0;

    // Create an object in the object store. Any metadata for this object must be
    // be passed in when the object is created.
    //
    // @param object_id The ID to use for the newly created object.
    // @param data_size The size in bytes of the space to be allocated for this
    //        object's data (this does not include space used for metadata).
    // @param device_num The number of the device where the object is being created.
    //        device_num = 0 corresponds to the host,
    //        device_num = 1 corresponds to GPU0,
    //        device_num = 2 corresponds to GPU1, etc.
    // @return The newly created object.
    //
    // The returned object must be released once it is done with.  It must also
    // be either sealed or aborted.

    virtual ObjectBufferPtr
    Create(const ObjectID &object_id,
           const int64_t data_size,
           const common::DeviceID &dev_id) = 0;

    // Create and seal an object in the object store. This is an optimization
    // which allows small objects to be created quickly with fewer messages to
    // the store.
    //
    // @param object_id The ID of the object to create.
    // @param data The data for the object to create.

    virtual void
    CreateAndSeal(const ObjectID &object_id,
                  const std::string &data) = 0;

    // Get some objects from the object store. This function will block until the
    // objects have all been created and sealed in the object store or the
    // timeout expires.
    //
    // If an object was not retrieved, the corresponding metadata and data
    // fields in the ObjectBuffer structure will evaluate to false.
    // Objects are automatically released by the client when their buffers
    // get out of scope.
    //
    // @param object_id The ID of the objects to get.
    // @param timeout_ms The amount of time in milliseconds to wait before this
    //        request times out. If this value is -1, then no timeout is set.
    // @return The object results.

    virtual ObjectBufferPtr
    Get(const ObjectID &object_id,
        int64_t timeout_ms = -1) = 0;

    // Get some objects from the object store. This function will block until the
    // objects have all been created and sealed in the object store or the
    // timeout expires.
    //
    // If an object was not retrieved, the corresponding metadata and data
    // fields in the ObjectBuffer structure will evaluate to false.
    // Objects are automatically released by the client when their buffers
    // get out of scope.
    //
    // @param object_ids The IDs of the objects to get.
    // @param timeout_ms The amount of time in milliseconds to wait before this
    //        request times out. If this value is -1, then no timeout is set.
    // @return The object results.

    virtual std::vector<ObjectBufferPtr>
    Get(const std::vector<ObjectID> &object_ids,
        int64_t timeout_ms = -1) = 0;

    // Tell object that the client no longer needs the object. This should be
    // called after Get() or Create() when the client is done with the object.
    // After this call, the buffer returned by Get() is no longer valid.
    //
    // @param object_id The ID of the object that is no longer needed.

    virtual void
    Release(const ObjectID &object_id) = 0;

    // Check if the object store contains a particular object and the object has
    // been sealed. The result will be stored in has_object.
    //
    // @param object_id The ID of the object whose presence we are checking.
    // @return The function will write true at this address if
    //         the object is present and false if it is not present.

    virtual bool
    Contains(const ObjectID &object_id) = 0;

    // Abort an unsealed object in the object store. If the abort succeeds, then
    // it will be as if the object was never created at all. The unsealed object
    // must have only a single reference (the one that would have been removed by
    // calling Seal).
    //
    // @param object_id The ID of the object to abort.

    virtual void
    Abort(const ObjectID &object_id) = 0;

    // Seal an object in the object store. The object will be immutable after
    // this call.
    //
    // @param object_id The ID of the object to seal.

    virtual void
    Seal(const ObjectID &object_id) = 0;

    // Subscribe object store's event.
    //
    // @param event_type the subscribed event type
    // @param callback the callback function registered for consuming the event.

    virtual void
    Subscribe(ObjectStoreEvent::Type event_type,
              std::function<void(ObjectStoreEventPtr)> callback) = 0;

    // Delete an object from the object store. This currently assumes that the
    // object is present, has been sealed and not used by another client. Otherwise,
    // it is a no operation.
    //
    // @param object_id The ID of the object to delete.

    virtual void
    Delete(const ObjectID &object_id) = 0;

    // Delete a list of objects from the object store. This currently assumes that the
    // object is present, has been sealed and not used by another client. Otherwise,
    // it is a no operation.
    //
    // @param object_ids The list of IDs of the objects to delete.

    virtual void
    Delete(const std::vector<ObjectID> &object_ids) = 0;

    // Disconnect from the local store instance

    virtual void
    Disconnect() = 0;
};


using ObjectStorePtr = std::shared_ptr<ObjectStore>;


} // namespace store
} // namespace zilliz
