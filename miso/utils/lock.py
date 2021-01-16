# import os
#
# import tempfile
# from miso.utils import singleton
#
#
# def try_get_lock():
#     try:
#         fn = os.path.join(tempfile.gettempdir(), "miso.lock")
#         with open(fn, 'w') as fh:
#             fh.write('miso')
#
#
#         try:
#             os.chmod(fn, 0o777)
#         except OSError:
#         pass
#
#     lock = singleton.SingleInstance(lockfile=fn)
#     print()
#     train_image_classification_model(tp)
#     done = True
#     except singleton.SingleInstanceException:
#     print(\{}: Another
#     script is already
#     running, trying
#     again in 10
#     seconds.({}
#     s
#     waiting)\\r\.format(datetime.now(), np.round(time.time() - start)), end = '')
#     time.sleep(10);
