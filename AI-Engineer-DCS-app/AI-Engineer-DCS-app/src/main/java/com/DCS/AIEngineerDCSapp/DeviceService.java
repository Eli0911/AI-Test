package com.DCS.AIEngineerDCSapp;

@Service
public class DeviceService {
    @Autowired
    private DeviceRepository deviceRepository;

    public List<Device> getAllDevices() {
        return deviceRepository.findAll();
    }

    public Device getDeviceById(Long id) {
        return deviceRepository.findById(id).orElse(null);
    }

    public Device addDevice(Device device) {
        return deviceRepository.save(device);
    }

    public boolean predictPrice(Long deviceId) {
        // Implement the logic to call the Python API and save the result in the device entity
        // Return true if the prediction is successful, false otherwise
    }
}